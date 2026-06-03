#include "scheduler2.hpp"

#include "core/logger.hpp"

#include <chrono>

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
