#include "backend_router.hpp"

#include <algorithm>
#include <cctype>

namespace powerserve {

namespace {

std::string normalize_backend(std::string backend_target) {
    std::transform(backend_target.begin(), backend_target.end(), backend_target.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return backend_target;
}

} // namespace

BackendKind BackendRouter::parse_backend_kind(const std::string &backend_target) {
    const std::string normalized = normalize_backend(backend_target);
    if (normalized == "cpu") {
        return BackendKind::CPU;
    }
    if (normalized == "npu") {
        return BackendKind::NPU;
    }
    return BackendKind::AUTO;
}

const char *BackendRouter::backend_name(BackendKind backend) {
    switch (backend) {
    case BackendKind::CPU:
        return "cpu";
    case BackendKind::NPU:
        return "npu";
    case BackendKind::AUTO:
    default:
        return "auto";
    }
}

BackendRouteDecision BackendRouter::route_for_generation_prefill(
    const std::string &backend_target,
    bool npu_available
) const {
    const BackendKind requested = parse_backend_kind(backend_target);
    if (requested == BackendKind::CPU) {
        return {.backend = BackendKind::CPU, .fallback = false, .note = ""};
    }
    if (requested == BackendKind::NPU) {
        if (npu_available) {
            return {.backend = BackendKind::NPU, .fallback = false, .note = ""};
        }
        return {.backend = BackendKind::CPU, .fallback = true, .note = "npu unavailable, fallback to cpu"};
    }

    if (npu_available) {
        return {.backend = BackendKind::NPU, .fallback = false, .note = "auto -> npu"};
    }
    return {.backend = BackendKind::CPU, .fallback = false, .note = "auto -> cpu"};
}

BackendRouteDecision BackendRouter::route_for_generation_decode(
    const std::string &backend_target,
    bool npu_available
) const {
    const BackendKind requested = parse_backend_kind(backend_target);
    if (requested == BackendKind::CPU) {
        return {.backend = BackendKind::CPU, .fallback = false, .note = ""};
    }
    if (requested == BackendKind::NPU) {
        if (npu_available) {
            return {.backend = BackendKind::NPU, .fallback = false, .note = ""};
        }
        return {.backend = BackendKind::CPU, .fallback = true, .note = "npu unavailable, fallback to cpu"};
    }

    // Keep current hetero intent: decode defaults to CPU in auto mode.
    return {.backend = BackendKind::CPU, .fallback = false, .note = "auto -> cpu"};
}

} // namespace powerserve
