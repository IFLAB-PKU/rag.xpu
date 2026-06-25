#include "scheduler2.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <mutex>

namespace powerserve {

namespace {

bool should_enqueue_cpu(BackendKind backend) {
    return backend == BackendKind::CPU || backend == BackendKind::AUTO;
}

// 4K profiling-derived estimates from the npu_cpu_cs run. Scores model the
// approximate remaining critical-path milliseconds from a node to DAG finish.
constexpr double kProfile4kIndexingMs = 5086.0;
constexpr double kProfile4kQueryExpandMs = 1115.0;
constexpr double kProfile4kQueryEmbeddingTotalMs = 1242.0;
constexpr double kProfile4kRetrievalBranchCount = 3.0;
constexpr double kProfile4kQueryEmbeddingNodeMs =
    kProfile4kQueryEmbeddingTotalMs / kProfile4kRetrievalBranchCount;
constexpr double kProfile4kSearchingMs = 16.0;
constexpr double kProfile4kRerankingMs = 4401.0;
constexpr double kProfile4kGenerationMs = 26973.0;
constexpr double kProfile4kGenerationPrefillMs = 2283.0;
constexpr double kProfile4kGenerationDecodeMs = 26179.0;

constexpr double kScoreGenerationMerge = 1.0;
constexpr double kScoreGenerationDecode = kProfile4kGenerationDecodeMs;
constexpr double kScoreGenerationPrefill = kProfile4kGenerationPrefillMs + kScoreGenerationDecode;
constexpr double kScoreReranking = kProfile4kRerankingMs + kProfile4kGenerationMs;
constexpr double kScoreSearching = kProfile4kSearchingMs + kScoreReranking;
constexpr double kScoreQueryEmbedding = kProfile4kQueryEmbeddingNodeMs + kScoreSearching;
constexpr double kScoreQueryExpand = kProfile4kQueryExpandMs + kProfile4kQueryEmbeddingTotalMs + kScoreSearching;
constexpr double kScoreIndexing = kProfile4kIndexingMs + kScoreSearching;

double critical_score_for_node(const Scheduler2DagNode &node) {
    if (!node.debug_name.empty()) {
        if (node.debug_name == "generation_merge") {
            return kScoreGenerationMerge;
        }
        if (node.debug_name.rfind("generation_decode_", 0) == 0) {
            return kScoreGenerationDecode;
        }
        if (node.debug_name.rfind("generation_prefill_", 0) == 0) {
            return kScoreGenerationPrefill;
        }
        if (node.debug_name == "reranking") {
            return kScoreReranking;
        }
        if (node.debug_name.rfind("searching_", 0) == 0) {
            return kScoreSearching;
        }
        if (node.debug_name.rfind("query_embedding_", 0) == 0) {
            return kScoreQueryEmbedding;
        }
        if (node.debug_name == "query_expand") {
            return kScoreQueryExpand;
        }
        if (node.debug_name == "indexing") {
            return kScoreIndexing;
        }
    }
    switch (node.type) {
    case Scheduler2TaskType::GENERATION_DECODE:
        return kScoreGenerationDecode;
    case Scheduler2TaskType::GENERATION_PREFILL:
        return kScoreGenerationPrefill;
    case Scheduler2TaskType::UNKNOWN:
    default:
        return 10.0;
    }
}

bool can_steal_task(BackendKind worker_backend, const Scheduler2Task &task) {
    if (task.type == Scheduler2TaskType::GENERATION_PREFILL ||
        task.type == Scheduler2TaskType::GENERATION_DECODE) {
        return task.backend == worker_backend;
    }
    return true;
}

} // namespace

Scheduler2::Scheduler2() :
    worker_cpu_([this]() { worker_loop(BackendKind::CPU); }),
    worker_npu_([this]() { worker_loop(BackendKind::NPU); }) {
    POWERSERVE_LOG_INFO("Scheduler2 started (FIFO, 2 workers: cpu+npu)");
}

Scheduler2::~Scheduler2() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
    if (worker_cpu_.joinable()) {
        worker_cpu_.join();
    }
    if (worker_npu_.joinable()) {
        worker_npu_.join();
    }
    POWERSERVE_LOG_INFO("Scheduler2 stopped");
}

void Scheduler2::enqueue_task(Scheduler2Task task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (should_enqueue_cpu(task.backend)) {
            auto insert_pos = cpu_queue_.end();
            for (auto iter = cpu_queue_.begin(); iter != cpu_queue_.end(); ++iter) {
                if (task.critical_score > iter->critical_score) {
                    insert_pos = iter;
                    break;
                }
            }
            cpu_queue_.insert(insert_pos, std::move(task));
        } else {
            auto insert_pos = npu_queue_.end();
            for (auto iter = npu_queue_.begin(); iter != npu_queue_.end(); ++iter) {
                if (task.critical_score > iter->critical_score) {
                    insert_pos = iter;
                    break;
                }
            }
            npu_queue_.insert(insert_pos, std::move(task));
        }
    }
    cv_.notify_all();
}

std::future<void> Scheduler2::submit_dag(std::vector<Scheduler2DagNode> nodes, bool enable_critical_score) {
    auto graph_promise = std::make_shared<std::promise<void>>();
    std::future<void> graph_future = graph_promise->get_future();

    if (nodes.empty()) {
        graph_promise->set_value();
        return graph_future;
    }

    auto node_map = std::make_shared<std::unordered_map<size_t, Scheduler2DagNode>>();
    node_map->reserve(nodes.size());
    for (auto &node : nodes) {
        if (node.fn == nullptr) {
            graph_promise->set_exception(std::make_exception_ptr(
                std::invalid_argument("scheduler2 dag node has empty function")
            ));
            return graph_future;
        }
        if (node_map->find(node.node_id) != node_map->end()) {
            graph_promise->set_exception(std::make_exception_ptr(
                std::invalid_argument("scheduler2 dag has duplicated node_id")
            ));
            return graph_future;
        }
        node_map->emplace(node.node_id, std::move(node));
    }

    if (enable_critical_score) {
        for (auto &[node_id, node] : *node_map) {
            node.critical_score = critical_score_for_node(node);
            POWERSERVE_LOG_DEBUG(
                "Scheduler2 critical_score init: node_id={}, name={}, type={}, backend={}, score={}",
                node_id,
                node.debug_name,
                scheduler2_task_type_name(node.type),
                BackendRouter::backend_name(node.backend),
                node.critical_score
            );
        }
    }

    auto indegree = std::make_shared<std::unordered_map<size_t, size_t>>();
    auto children = std::make_shared<std::unordered_map<size_t, std::vector<size_t>>>();
    auto graph_mutex = std::make_shared<std::mutex>();
    indegree->reserve(node_map->size());
    children->reserve(node_map->size());

    for (const auto &[node_id, node] : *node_map) {
        (void)node;
        indegree->emplace(node_id, 0);
        children->emplace(node_id, std::vector<size_t>{});
    }

    for (const auto &[node_id, node] : *node_map) {
        size_t &deg = indegree->at(node_id);
        for (const size_t dep : node.dependencies) {
            auto dep_iter = node_map->find(dep);
            if (dep_iter == node_map->end()) {
                graph_promise->set_exception(std::make_exception_ptr(
                    std::invalid_argument("scheduler2 dag dependency node_id not found")
                ));
                return graph_future;
            }
            children->at(dep).push_back(node_id);
            ++deg;
        }
    }

    std::deque<size_t> topo_queue;
    topo_queue.clear();
    for (const auto &[node_id, deg] : *indegree) {
        if (deg == 0) {
            topo_queue.push_back(node_id);
        }
    }
    const size_t zero_indegree_count = topo_queue.size();
    if (zero_indegree_count == 0) {
        graph_promise->set_exception(std::make_exception_ptr(
            std::invalid_argument("scheduler2 dag has no zero-indegree node (possible cycle)")
        ));
        return graph_future;
    }
    auto indegree_check = *indegree;
    size_t topo_visited = 0;
    while (!topo_queue.empty()) {
        const size_t current = topo_queue.front();
        topo_queue.pop_front();
        ++topo_visited;
        for (const size_t child : children->at(current)) {
            size_t &child_deg = indegree_check[child];
            if (child_deg > 0) {
                --child_deg;
                if (child_deg == 0) {
                    topo_queue.push_back(child);
                }
            }
        }
    }
    if (topo_visited != node_map->size()) {
        graph_promise->set_exception(std::make_exception_ptr(
            std::invalid_argument("scheduler2 dag contains cycle")
        ));
        return graph_future;
    }

    auto completed = std::make_shared<std::atomic_size_t>(0);
    auto failed = std::make_shared<std::atomic_bool>(false);
    auto finished = std::make_shared<std::atomic_bool>(false);
    auto fail_mutex = std::make_shared<std::mutex>();
    auto first_error = std::make_shared<std::exception_ptr>();
    auto enqueue_order = std::make_shared<std::vector<std::string>>();
    auto enqueue_order_mutex = std::make_shared<std::mutex>();
    auto log_enqueue_order = std::make_shared<std::function<void(const char *)>>();
    *log_enqueue_order = [enqueue_order, enqueue_order_mutex](const char *status) {
        std::ostringstream summary;
        {
            std::lock_guard<std::mutex> lock(*enqueue_order_mutex);
            for (size_t i = 0; i < enqueue_order->size(); ++i) {
                if (i > 0) {
                    summary << " -> ";
                }
                summary << (*enqueue_order)[i];
            }
        }
        POWERSERVE_LOG_INFO("Scheduler2 critical order summary ({}): {}", status, summary.str());
    };

    auto enqueue_child = std::make_shared<std::function<void(size_t)>>();
    *enqueue_child = [this,
                      node_map,
                      indegree,
                      children,
                      graph_mutex,
                      completed,
                      failed,
                      finished,
                      graph_promise,
                      fail_mutex,
                      first_error,
                      enable_critical_score,
                      enqueue_order,
                      enqueue_order_mutex,
                      log_enqueue_order,
                      enqueue_child](size_t node_id) {
        const auto node_iter = node_map->find(node_id);
        if (node_iter == node_map->end()) {
            return;
        }
        const Scheduler2DagNode &node = node_iter->second;
        if (enable_critical_score) {
            POWERSERVE_LOG_DEBUG(
                "Scheduler2 enqueue node: name={}, node_id={}, type={}, backend={}, critical_score={}",
                node.debug_name,
                node.node_id,
                scheduler2_task_type_name(node.type),
                BackendRouter::backend_name(node.backend),
                node.critical_score
            );
            std::ostringstream item;
            item << node.debug_name
                 << "#id=" << node.node_id
                 << "#backend=" << BackendRouter::backend_name(node.backend)
                 << "#score=" << node.critical_score;
            std::lock_guard<std::mutex> lock(*enqueue_order_mutex);
            enqueue_order->push_back(item.str());
        }
        this->enqueue_task(Scheduler2Task{
            .type = node.type,
            .request_id = node.request_id,
            .backend = node.backend,
            .fn = [node_id,
                   node_map,
                   indegree,
                   children,
                   graph_mutex,
                   completed,
                   failed,
                   finished,
                   graph_promise,
                   fail_mutex,
                   first_error,
                   enable_critical_score,
                   log_enqueue_order,
                   enqueue_child]() {
                if (failed->load()) {
                    return;
                }

                try {
                    node_map->at(node_id).fn();
                } catch (...) {
                    bool expected = false;
                    if (failed->compare_exchange_strong(expected, true)) {
                        std::lock_guard<std::mutex> lock(*fail_mutex);
                        *first_error = std::current_exception();
                    }
                }

                if (failed->load()) {
                    if (!finished->exchange(true)) {
                        if (enable_critical_score) {
                            (*log_enqueue_order)("failed");
                        }
                        std::exception_ptr err;
                        {
                            std::lock_guard<std::mutex> lock(*fail_mutex);
                            err = *first_error;
                        }
                        if (err == nullptr) {
                            err = std::make_exception_ptr(
                                std::runtime_error("scheduler2 dag failed without exception")
                            );
                        }
                        graph_promise->set_exception(err);
                    }
                    return;
                }

                const size_t done = completed->fetch_add(1) + 1;
                if (done == node_map->size()) {
                    if (!finished->exchange(true)) {
                        if (enable_critical_score) {
                            (*log_enqueue_order)("completed");
                        }
                        graph_promise->set_value();
                    }
                    return;
                }

                std::vector<size_t> ready_children;
                {
                    std::lock_guard<std::mutex> lock(*graph_mutex);
                    for (const size_t child : children->at(node_id)) {
                        size_t &child_deg = indegree->at(child);
                        if (child_deg > 0) {
                            --child_deg;
                            if (child_deg == 0) {
                                ready_children.push_back(child);
                            }
                        }
                    }
                }
                if (enable_critical_score) {
                    std::sort(ready_children.begin(), ready_children.end(), [node_map](size_t lhs, size_t rhs) {
                        const auto &left = node_map->at(lhs);
                        const auto &right = node_map->at(rhs);
                        if (left.critical_score != right.critical_score) {
                            return left.critical_score > right.critical_score;
                        }
                        return left.node_id < right.node_id;
                    });
                }
                for (const size_t child : ready_children) {
                    (*enqueue_child)(child);
                }
            },
            .enqueued_at = std::chrono::steady_clock::now(),
        });
    };

    std::vector<size_t> root_nodes;
    for (const auto &[node_id, deg] : *indegree) {
        if (deg == 0) {
            root_nodes.push_back(node_id);
        }
    }
    if (enable_critical_score) {
        std::sort(root_nodes.begin(), root_nodes.end(), [node_map](size_t lhs, size_t rhs) {
            const auto &left = node_map->at(lhs);
            const auto &right = node_map->at(rhs);
            if (left.critical_score != right.critical_score) {
                return left.critical_score > right.critical_score;
            }
            return left.node_id < right.node_id;
        });
        for (const size_t root_node_id : root_nodes) {
            const auto &node = node_map->at(root_node_id);
            POWERSERVE_LOG_DEBUG(
                "Scheduler2 root node candidate: name={}, node_id={}, type={}, backend={}, critical_score={}",
                node.debug_name,
                node.node_id,
                scheduler2_task_type_name(node.type),
                BackendRouter::backend_name(node.backend),
                node.critical_score
            );
        }
    }
    for (const size_t root_node_id : root_nodes) {
        (*enqueue_child)(root_node_id);
    }

    POWERSERVE_LOG_INFO("Scheduler2 submit dag: nodes={}, roots={}", node_map->size(), zero_indegree_count);
    return graph_future;
}

void Scheduler2::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    drain_cv_.wait(lock, [this]() { return cpu_queue_.empty() && npu_queue_.empty() && active_count_ == 0; });
}

size_t Scheduler2::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_queue_.size() + npu_queue_.size();
}

size_t Scheduler2::active_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_count_;
}

void Scheduler2::worker_loop(BackendKind worker_backend) {
    const char *worker_name = (worker_backend == BackendKind::NPU) ? "npu" : "cpu";
    while (true) {
        Scheduler2Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this, worker_backend]() {
                if (shutdown_) {
                    return true;
                }
                if (worker_backend == BackendKind::CPU) {
                    return !cpu_queue_.empty() ||
                        (!npu_queue_.empty() && can_steal_task(worker_backend, npu_queue_.front()));
                }
                return !npu_queue_.empty() ||
                    (!cpu_queue_.empty() && can_steal_task(worker_backend, cpu_queue_.front()));
            });

            if (shutdown_ && cpu_queue_.empty() && npu_queue_.empty()) {
                break;
            }

            auto pop_from_queue = [](std::deque<Scheduler2Task> &queue, Scheduler2Task &out) -> bool {
                if (queue.empty()) {
                    return false;
                }
                out = std::move(queue.front());
                queue.pop_front();
                return true;
            };

            bool popped = false;
            if (worker_backend == BackendKind::CPU) {
                popped = pop_from_queue(cpu_queue_, task);
                if (!popped && !npu_queue_.empty() && can_steal_task(worker_backend, npu_queue_.front())) {
                    popped = pop_from_queue(npu_queue_, task); // steal
                }
            } else {
                popped = pop_from_queue(npu_queue_, task);
                if (!popped && !cpu_queue_.empty() && can_steal_task(worker_backend, cpu_queue_.front())) {
                    popped = pop_from_queue(cpu_queue_, task); // steal
                }
            }
            if (!popped) {
                continue;
            }
            ++active_count_;
        }

        const auto start = std::chrono::steady_clock::now();
        try {
            if (task.fn) {
                task.fn();
            }
        } catch (...) {
            // Exception is captured by packaged_task future.
        }
        const auto end = std::chrono::steady_clock::now();
        const size_t queue_wait_ms = static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(start - task.enqueued_at).count()
        );
        const size_t exec_ms = static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        );
        POWERSERVE_LOG_DEBUG(
            "Scheduler2 task done: worker={}, type={}, request_id={}, backend={}, queue_wait_ms={}, exec_ms={}",
            worker_name,
            scheduler2_task_type_name(task.type),
            task.request_id,
            BackendRouter::backend_name(task.backend),
            queue_wait_ms,
            exec_ms
        );

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_count_;
        }
        drain_cv_.notify_all();
    }
}

} // namespace powerserve
