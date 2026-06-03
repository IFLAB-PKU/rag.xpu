#pragma once

#include <string>

namespace powerserve {

enum class BackendKind {
    AUTO,
    CPU,
    NPU,
};

struct BackendRouteDecision {
    BackendKind backend = BackendKind::CPU;
    bool fallback = false;
    std::string note;
};

class BackendRouter {
public:
    static BackendKind parse_backend_kind(const std::string &backend_target);
    static const char *backend_name(BackendKind backend);

    BackendRouteDecision route_for_generation_prefill(const std::string &backend_target, bool npu_available) const;
    BackendRouteDecision route_for_generation_decode(const std::string &backend_target, bool npu_available) const;
};

} // namespace powerserve
