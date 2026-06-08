#pragma once

#include "backend_router.hpp"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace powerserve {

enum class Scheduler2TaskType {
    UNKNOWN,
    GENERATION_PREFILL,
    GENERATION_DECODE,
};

inline const char *scheduler2_task_type_name(Scheduler2TaskType type) {
    switch (type) {
    case Scheduler2TaskType::GENERATION_PREFILL:
        return "GENERATION_PREFILL";
    case Scheduler2TaskType::GENERATION_DECODE:
        return "GENERATION_DECODE";
    case Scheduler2TaskType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

struct Scheduler2Task {
    Scheduler2TaskType type = Scheduler2TaskType::UNKNOWN;
    size_t request_id = 0;
    BackendKind backend = BackendKind::CPU;
    std::function<void()> fn;
    std::chrono::steady_clock::time_point enqueued_at = std::chrono::steady_clock::now();
};

struct Scheduler2DagNode {
    size_t node_id = 0;
    Scheduler2TaskType type = Scheduler2TaskType::UNKNOWN;
    size_t request_id = 0;
    BackendKind backend = BackendKind::CPU;
    std::vector<size_t> dependencies;
    std::function<void()> fn;
    std::string debug_name;
};

class Scheduler2 {
public:
    Scheduler2();
    ~Scheduler2();

    template <typename F>
    auto submit(Scheduler2TaskType type, size_t request_id, BackendKind backend, F &&fn) -> std::future<decltype(fn())> {
        using ReturnType = decltype(fn());

        auto packaged = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(fn));
        std::future<ReturnType> result = packaged->get_future();

        enqueue_task(Scheduler2Task{
            .type = type,
            .request_id = request_id,
            .backend = backend,
            .fn = [packaged]() { (*packaged)(); },
            .enqueued_at = std::chrono::steady_clock::now(),
        });
        return result;
    }

    std::future<void> submit_dag(std::vector<Scheduler2DagNode> nodes);
    void drain();
    size_t pending_count() const;
    size_t active_count() const;

private:
    std::deque<Scheduler2Task> cpu_queue_;
    std::deque<Scheduler2Task> npu_queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drain_cv_;
    std::thread worker_cpu_;
    std::thread worker_npu_;
    bool shutdown_ = false;
    size_t active_count_ = 0;

    void enqueue_task(Scheduler2Task task);
    void worker_loop(BackendKind worker_backend);
};

} // namespace powerserve
