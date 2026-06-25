#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary_path="${1:-/tmp/thread_pool_shutdown_race}"
iterations="${2:-32}"

c++ \
    -std=c++20 \
    -O2 \
    -pthread \
    -DFMT_HEADER_ONLY \
    -I"$repo_root/src" \
    -I"$repo_root/libs/fmt/include" \
    "$repo_root/tests/thread_pool_shutdown_race.cpp" \
    "$repo_root/src/core/thread_pool.cpp" \
    "$repo_root/src/core/spin_barrier.cpp" \
    -o "$binary_path"

timeout 5s "$binary_path" "$iterations"
