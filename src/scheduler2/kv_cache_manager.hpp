#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace powerserve {

struct KvCacheRecord {
    size_t request_id = 0;
    std::string model_id;
    std::string producer_backend;
    size_t kv_begin = 0;
    size_t kv_end = 0;
    size_t prefill_tokens_total = 0;
    std::string bridge_status = "idle";
    size_t bridge_cost_ms = 0;
};

class KvCacheManager {
public:
    void put(const KvCacheRecord &record);
    std::optional<KvCacheRecord> get(size_t request_id) const;
    bool bridge_to_cpu(size_t request_id);
    void release(size_t request_id);
    bool empty() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<size_t, KvCacheRecord> records_;
};

} // namespace powerserve
