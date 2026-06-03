#include "scheduler2.hpp"

#include "core/logger.hpp"

#include <chrono>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace powerserve {

Scheduler2::Scheduler2() : worker_([this]() { worker_loop(); }) {
    POWERSERVE_LOG_INFO("Scheduler2 started (FIFO, 1 worker)");
}

Scheduler2::~Scheduler2() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    POWERSERVE_LOG_INFO("Scheduler2 stopped");
}

void Scheduler2::enqueue_task(Scheduler2Task task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(task));
    }
    cv_.notify_one();
}

std::future<void> Scheduler2::submit_dag(std::vector<Scheduler2DagNode> nodes) {
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

    auto indegree = std::make_shared<std::unordered_map<size_t, size_t>>();
    auto children = std::make_shared<std::unordered_map<size_t, std::vector<size_t>>>();
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

    auto enqueue_ready = std::make_shared<std::function<void(size_t)>>();
    *enqueue_ready = [this,
                      node_map,
                      indegree,
                      children,
                      completed,
                      failed,
                      finished,
                      graph_promise,
                      fail_mutex,
                      first_error,
                      enqueue_ready](size_t node_id) {
        const auto node_iter = node_map->find(node_id);
        if (node_iter == node_map->end()) {
            return;
        }
        const Scheduler2DagNode &node = node_iter->second;
        this->enqueue_task(Scheduler2Task{
            .type = node.type,
            .request_id = node.request_id,
            .backend = node.backend,
            .fn = [node_id,
                   node_map,
                   indegree,
                   children,
                   completed,
                   failed,
                   finished,
                   graph_promise,
                   fail_mutex,
                   first_error,
                   enqueue_ready]() {
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
                        graph_promise->set_value();
                    }
                    return;
                }

                for (const size_t child : children->at(node_id)) {
                    size_t &child_deg = indegree->at(child);
                    if (child_deg > 0) {
                        --child_deg;
                        if (child_deg == 0) {
                            (*enqueue_ready)(child);
                        }
                    }
                }
            },
            .enqueued_at = std::chrono::steady_clock::now(),
        });
    };

    for (const auto &[node_id, deg] : *indegree) {
        if (deg == 0) {
            (*enqueue_ready)(node_id);
        }
    }

    POWERSERVE_LOG_INFO("Scheduler2 submit dag: nodes={}, roots={}", node_map->size(), zero_indegree_count);
    return graph_future;
}

void Scheduler2::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    drain_cv_.wait(lock, [this]() { return queue_.empty() && active_count_ == 0; });
}

size_t Scheduler2::pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

size_t Scheduler2::active_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_count_;
}

void Scheduler2::worker_loop() {
    while (true) {
        Scheduler2Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return shutdown_ || !queue_.empty(); });

            if (shutdown_ && queue_.empty()) {
                break;
            }

            task = std::move(queue_.front());
            queue_.pop_front();
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
            "Scheduler2 task done: type={}, request_id={}, backend={}, queue_wait_ms={}, exec_ms={}",
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
