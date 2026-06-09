// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "backend/ggml/ggml_kv_cache.hpp"

#include "backend/cpu_buffer.hpp"

namespace powerserve::ggml {

GGMLKV::GGMLKV(const ModelConfig::LLMConfig &config) :
    m_kv_dim(config.kv_dim),
    m_n_kv_heads(config.n_kv_heads),
    m_n_ctx(config.seq_len),
    m_n_layers(config.n_layers),
    m_head_size(config.head_size),
    m_batch_size(1), // FIXME:
    m_config(config) {

    prepare_model_chunk();

    kv_cache = std::make_unique<KVCache<GGMLKVInterface>>(m_n_layers, m_n_kv_heads, m_n_ctx, *this, chunk);
}

void GGMLKV::prepare_model_chunk() {
    auto &key_buffer   = chunk.key_buffer;
    auto &value_buffer = chunk.value_buffer;
    auto &k            = chunk.current_k;
    auto &v            = chunk.current_v;

    key_buffer.resize(m_n_layers);
    value_buffer.resize(m_n_layers);
    size_t layer_size = m_kv_dim * m_n_ctx;
    for (size_t L = 0; L < m_n_layers; L++) {
        key_buffer[L].reserve(layer_size);
        value_buffer[L].reserve(layer_size);

        chunk.key_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
        chunk.value_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
        Stride stride = {
            sizeof(float),
            sizeof(float) * m_n_ctx,
            sizeof(float) * m_kv_dim * m_n_ctx,
            sizeof(float) * m_kv_dim * m_n_ctx
        };
        chunk.key_tensors[L].m_data   = std::make_shared<CPUBuffer>(stride, key_buffer[L].data());
        chunk.value_tensors[L].m_data = std::make_shared<CPUBuffer>(stride, value_buffer[L].data());
    }

    k.resize(m_n_layers);
    v.resize(m_n_layers);
    for (size_t L = 0; L < m_n_layers; L++) {
        k[L].reserve(m_batch_size * m_kv_dim);
        v[L].reserve(m_batch_size * m_kv_dim);
    }

    auto &attn_bias = chunk.attn_bias;
    attn_bias.reserve(m_batch_size * m_n_ctx);
}

auto GGMLKV::save_snapshot() const -> std::unique_ptr<Snapshot> {
    return save_snapshot(0, kv_cache->position);
}

auto GGMLKV::save_snapshot(size_t begin_position, size_t end_position) const -> std::unique_ptr<Snapshot> {
    POWERSERVE_ASSERT(begin_position <= end_position);
    POWERSERVE_ASSERT(end_position <= kv_cache->position);
    POWERSERVE_ASSERT(end_position <= m_n_ctx);

    auto snapshot = std::make_unique<Snapshot>();
    snapshot->begin_position = begin_position;
    // For partial snapshots, `position` is the end of the captured interval.
    // Full snapshot still uses [0, kv_cache->position) via save_snapshot().
    snapshot->position = end_position;
    snapshot->key_buffer.resize(m_n_layers);
    snapshot->value_buffer.resize(m_n_layers);
    const size_t token_count = end_position - begin_position;
    const size_t key_layer_size = token_count * m_kv_dim;
    const size_t value_layer_size = token_count * m_kv_dim;
    for (size_t L = 0; L < m_n_layers; L++) {
        auto &dst_key = snapshot->key_buffer[L];
        auto &dst_value = snapshot->value_buffer[L];
        dst_key.resize(key_layer_size);
        dst_value.resize(value_layer_size);
        if (token_count == 0) {
            continue;
        }

        // key_buffer layout is token-major contiguous: [token][kv_dim]
        const float *key_src = chunk.key_buffer[L].data() + begin_position * m_kv_dim;
        std::memcpy(dst_key.data(), key_src, key_layer_size * sizeof(float));

        // value_buffer layout is [head][head_size][n_ctx], token dimension is strided by n_ctx.
        // Copy through KV views to preserve the exact memory layout.
        size_t dst_flat = 0;
        for (size_t token = begin_position; token < end_position; ++token) {
            for (size_t head = 0; head < m_n_kv_heads; ++head) {
                const auto view = kv_cache->value_entry({
                    .layer_id = L,
                    .head_id = head,
                    .index = token,
                });
                POWERSERVE_ASSERT(view.element_size == sizeof(float));
                for (size_t i = 0; i < view.n_elements; ++i) {
                    const auto *src_ptr = reinterpret_cast<const float *>(
                        static_cast<const char *>(view.data) + i * view.stride
                    );
                    dst_value[dst_flat++] = *src_ptr;
                }
            }
        }
        POWERSERVE_ASSERT(dst_flat == value_layer_size);
    }
    return snapshot;
}

void GGMLKV::restore_snapshot(const Snapshot &snapshot) {
    POWERSERVE_ASSERT(snapshot.begin_position <= snapshot.position);
    POWERSERVE_ASSERT(snapshot.position <= m_n_ctx);
    POWERSERVE_ASSERT(snapshot.key_buffer.size() == m_n_layers);
    POWERSERVE_ASSERT(snapshot.value_buffer.size() == m_n_layers);
    const size_t token_count = snapshot.position - snapshot.begin_position;
    const size_t key_layer_size = token_count * m_kv_dim;
    const size_t value_layer_size = token_count * m_kv_dim;
    for (size_t L = 0; L < m_n_layers; L++) {
        const auto &src_key = snapshot.key_buffer[L];
        const auto &src_value = snapshot.value_buffer[L];
        POWERSERVE_ASSERT(src_key.size() == key_layer_size);
        POWERSERVE_ASSERT(src_value.size() == value_layer_size);
        if (token_count == 0) {
            continue;
        }

        // key_buffer layout is token-major contiguous: [token][kv_dim]
        float *key_dst = chunk.key_buffer[L].data() + snapshot.begin_position * m_kv_dim;
        std::memcpy(key_dst, src_key.data(), key_layer_size * sizeof(float));

        // value_buffer layout is [head][head_size][n_ctx], token dimension is strided by n_ctx.
        // Restore through KV views to keep the layout correct.
        size_t src_flat = 0;
        for (size_t token = snapshot.begin_position; token < snapshot.position; ++token) {
            for (size_t head = 0; head < m_n_kv_heads; ++head) {
                const auto view = kv_cache->value_entry({
                    .layer_id = L,
                    .head_id = head,
                    .index = token,
                });
                POWERSERVE_ASSERT(view.element_size == sizeof(float));
                for (size_t i = 0; i < view.n_elements; ++i) {
                    auto *dst_ptr = reinterpret_cast<float *>(
                        static_cast<char *>(view.data) + i * view.stride
                    );
                    *dst_ptr = src_value[src_flat++];
                }
            }
        }
        POWERSERVE_ASSERT(src_flat == value_layer_size);
    }
    kv_cache->position = snapshot.position;
}

} // namespace powerserve::ggml
