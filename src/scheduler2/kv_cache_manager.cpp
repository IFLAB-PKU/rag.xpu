#include "kv_cache_manager.hpp"

namespace powerserve {

void KvCacheManager::put(const KvCacheRecord &record) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_[record.request_id] = record;
}

std::optional<KvCacheRecord> KvCacheManager::get(size_t request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = records_.find(request_id);
    if (iter == records_.end()) {
        return std::nullopt;
    }
    return iter->second;
}

bool KvCacheManager::bridge_to_cpu(size_t request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.find(request_id) != records_.end();
}

void KvCacheManager::release(size_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.erase(request_id);
}

bool KvCacheManager::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_.empty();
}

} // namespace powerserve
