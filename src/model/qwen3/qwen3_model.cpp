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

#include "qwen3_model.hpp"

#include "backend/cpu_buffer.hpp"
#include "core/logger.hpp"
#include "core/perfetto_trace.hpp"
#include "executor/executor.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "model/qwen3/qwen3_weight.hpp"
#include "sampler/sampler.hpp"
#include "tokenizer/tokenizer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace powerserve {

Qwen3Model::Qwen3Model(const std::string &filename, const std::shared_ptr<ModelConfig> &config) : Model(filename) {
    {
        gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
        gguf_ctx                = gguf_init_from_file(filename.c_str(), params);
        POWERSERVE_ASSERT(gguf_ctx != nullptr);
        POWERSERVE_ASSERT(ggml_ctx != nullptr);
    }
    m_config  = config;
    lazy_load = ggml_get_tensor(ggml_ctx, "output_norm.weight") == nullptr ? true : false;
    m_weights = std::make_shared<Qwen3Weight>(ggml_ctx, m_config->llm.n_layers, lazy_load);
    if (lazy_load) {
        POWERSERVE_LOG_WARN("only the embedding table was loaded");
    }
    m_ffn = std::make_shared<FFN>(m_config->llm, m_weights);
}

Qwen3Model::~Qwen3Model() {
    gguf_free(gguf_ctx);
}

auto Qwen3Model::forward(
    const std::vector<int> &tokens, const std::vector<int> &pos, const CausalAttentionMask &mask, bool lm_head
) -> LogitsVector {
    Graph g(m_config->model_id);
    // input embedding
    size_t batch_size  = tokens.size();
    auto embd_tb       = g.add_tensor(m_weights->token_embedding_table);
    auto x             = g.get_embedding(embd_tb, tokens);
    TensorNode *logits = nullptr;

    auto &llm_config = m_config->llm;

#if defined(POWERSERVE_WITH_QNN)
    if (m_platform->qnn_backend) {
        auto size            = llm_config.dim;
        bool use_qnn_lm_head = m_platform->qnn_backend->m_models[m_config->model_id]->m_config.lm_heads.size() > 0;
        if (use_qnn_lm_head) {
            size   = llm_config.vocab_size;
            logits = g.qnn_forward(x, pos, mask, size, lm_head);
        } else {
            x = g.qnn_forward(x, pos, mask, size, lm_head);
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    } else
#endif
    {
        if (!lazy_load) {
            m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(batch_size);
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                /* lsh 修改起始 */
                bool is_need_bias = false;
                /* lsh 修改结束 */
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), pos, mask, is_need_bias); //Qwen2最后一个参数硬编码为True，要求使用bias
                auto ffn_o = m_ffn->build(g, att_o, L);
                x          = ffn_o;
            }
            // TODO: cpu and qnn reuse
            if (lm_head) {
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                auto output_w       = g.add_tensor(m_weights->output_weight);
                logits              = g.mat_mul(output_w, final_rms_norm);
            }
        }
    }

    Executor executor(*m_platform, g);
    executor.allocate_buffers();

    executor.run();
#if defined(POWERSERVE_WITH_QNN)
    if (!m_platform->qnn_backend)
#endif
    {
        m_platform->ggml_backends[m_config->model_id]->m_kv->advance(batch_size);
    }

    if (!lm_head) {
        return LogitsVector();
    }

    return LogitsVector(logits->m_data, m_config->llm.vocab_size, batch_size);
}

auto Qwen3Model::decode(Sampler &sampler, const std::vector<Token> tokens, const std::vector<int> pos, bool lm_head)
    -> std::vector<Token> {
    auto mask = CausalAttentionMask(tokens.size());
    auto ret  = forward(tokens, pos, mask, lm_head);
    std::vector<Token> toks;
    for (auto logits : ret.logits_vector) {
        auto probs = ProbArray(logits);
        sampler.apply(probs);
        auto next = probs.greedy_sample().token;
        sampler.accept(next);
        toks.push_back(next);
    }
    return toks;
}

auto Qwen3Model::generate(
    const Tokenizer &tokenizer, Sampler &sampler, const std::string &prompt, int steps, size_t batch_size
) -> std::shared_ptr<TokenIterator> {
    return std::make_shared<ModelTokenIterator>(*this, tokenizer, sampler, prompt, steps, batch_size);
}

// `compute_embedding` does not interact with ModelIterator, so it prepares its environment, say `m_platform` and `pos`, in itself.
auto Qwen3Model::compute_embedding(const std::vector<Token> &tokens, size_t batch_size)
    -> std::vector<float> {
    auto &llm_config = m_config->llm;
    size_t n_tokens  = tokens.size();

    // Set up platform
    m_platform->reset_kv_position(m_config->model_id);
    m_platform->ggml_backends[m_config->model_id]->setup_threadpool();

    size_t n_processed = 0;

    // Loop processing tokens in batches. 
    while(n_processed < n_tokens) {
        size_t current_bs = std::min(batch_size, n_tokens - n_processed);
        std::vector<int> batch_tokens(tokens.begin() + n_processed, tokens.begin() + n_processed + current_bs);

        int start_pos = m_platform->get_kv_position(m_config->model_id);
        std::vector<int> batch_pos(current_bs);
        std::iota(batch_pos.begin(), batch_pos.end(), start_pos);

        auto mask = CausalAttentionMask(current_bs);
        bool is_last_batch = (n_processed + current_bs == n_tokens); // identify last token

        // build graph
        Graph g(m_config->model_id);
        // input embedding
        auto embd_tb = g.add_tensor(m_weights->token_embedding_table);
        auto x       = g.get_embedding(embd_tb, batch_tokens);

#if defined(POWERSERVE_WITH_QNN)
        if (m_platform->qnn_backend) {
            auto size = llm_config.dim;
            bool use_qnn_lm_head = m_platform->qnn_backend->m_models[m_config->model_id]->m_config.lm_heads.size() > 0;
            
            if (use_qnn_lm_head) {
                size   = llm_config.vocab_size;
                x = g.qnn_forward(x, batch_pos, mask, size, false /* lm_head */);
            }else{
                // 调用 qnn_forward
                // 注意：这里假设 qnn_forward 能正确处理 lm_head=false 的情况并返回 hidden states
                x = g.qnn_forward(x, batch_pos, mask, size, false /* lm_head */);

                // 如果是最后一批，且 QNN 输出的已经是 hidden state，
                // 我们可能还需要手动加上 Final RMS Norm，除非 QNN 模型内部已经包含了它。
                // 通常 QNN 模型导出时会包含所有层，但不一定包含最后的 Norm。
                // 参考 forward 中的逻辑：
                // 对于 Embedding，我们需要 Final RMS Norm。
                // 如果 QNN Graph 仅仅是 Transformer Layers 的堆叠，我们需要在这里补上 Norm。
                if (is_last_batch) {
                    auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                    auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                    x                   = final_rms_norm;
                }
            }
            
        } else
#endif
        if(!lazy_load){
            m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(current_bs); // use REAL batch size
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                bool is_need_bias = false;
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), batch_pos, mask, is_need_bias);
                auto ffn_o = m_ffn->build(g, att_o, L);
                x          = ffn_o;
            }
            
            if(is_last_batch) { // Qwen3-Embedding uses "last token pooling"
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                x                   = final_rms_norm;
            }
        }

        // Execute the graph
        Executor executor(*m_platform, g);
        executor.allocate_buffers();
        executor.run();

    #if defined(POWERSERVE_WITH_QNN)
        if (!m_platform->qnn_backend)
    #endif
        {
            m_platform->ggml_backends[m_config->model_id]->m_kv->advance(current_bs);
        }

        if (is_last_batch){
            const size_t dim = llm_config.dim; //embd_dim, see src/core/config.cpp
            EmbeddingVector view(x->m_data, dim, current_bs);
            auto last_token_span = view.embeddings.back();

            std::vector<float> embedding(dim);
            float norm_sq = 0.0f;

            for (size_t i = 0; i < dim; ++i) {
                float val = last_token_span[i];
                embedding[i] = val;
                norm_sq += val * val;
            }

            // L2 Norm
            float norm = std::sqrt(norm_sq);
            const float eps = 1e-12f;
            if (norm > eps) {
                for (size_t i = 0; i < dim; ++i) {
                    embedding[i] /= norm;
                }
            }
            
            // clean thread pool, see ~ModelTokenIterator() in src/model/model.hpp
            m_platform->ggml_backends[m_config->model_id]->reset_threadpool();
            
            return embedding;
        }
        n_processed += current_bs;
    }
    return {}; //should not reach here
} 

auto Qwen3Model::compute_rerank_score(const std::vector<Token> &tokens, size_t batch_size) -> float {
    auto &llm_config = m_config->llm;
    size_t n_tokens  = tokens.size();

    // 1. Setup Environment
    m_platform->reset_kv_position(m_config->model_id);
    m_platform->ggml_backends[m_config->model_id]->setup_threadpool();

    size_t n_processed = 0;

    // 2. Loop processing tokens in batches (Mimics compute_embedding)
    while (n_processed < n_tokens) {
        size_t current_bs = std::min(batch_size, n_tokens - n_processed);
        std::vector<int> batch_tokens(tokens.begin() + n_processed, tokens.begin() + n_processed + current_bs);

        int start_pos = m_platform->get_kv_position(m_config->model_id);
        std::vector<int> batch_pos(current_bs);
        std::iota(batch_pos.begin(), batch_pos.end(), start_pos);

        auto mask = CausalAttentionMask(current_bs);
        bool is_last_batch = (n_processed + current_bs == n_tokens);

        // 3. Build Graph
        Graph g(m_config->model_id);
        auto embd_tb = g.add_tensor(m_weights->token_embedding_table);
        auto x       = g.get_embedding(embd_tb, batch_tokens);

#if defined(POWERSERVE_WITH_QNN)
        if (m_platform->qnn_backend) {
            // NPU Backbone: Use QNN to calculate hidden states
            // This corresponds to your qwen3_reranker_0.6b_x.bin binaries
            auto size = llm_config.dim;
            x = g.qnn_forward(x, batch_pos, mask, size, true /* lm_head=true */);
            // [NOTE]: `lm_head=true` in order to satisfy qnn_forward's output size. Indeed we do not use lm_head.bin to compute logits. We want hidden states.

            // CPU Tail: Manually attach the Rerank Head logic
            // We intentionally ignore the QNN's lm_head.bin
            // and use the precise valid metrics from GGUF (cls.output.weight 2x1024)
            if (is_last_batch) {
                // 1. Final RMS Norm (CPU)
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                
                // 2. Output Projection to [Batch, 2] (CPU)
                auto output_w       = g.add_tensor(m_weights->output_weight);
                x                   = g.mat_mul(output_w, final_rms_norm);
                
                // 3. Softmax (CPU)
                x                   = g.softmax(x);
            }
        } else
#endif
        if (!lazy_load) {
            m_platform->ggml_backends[m_config->model_id]->reset_kv_batch_size(current_bs);
            for (size_t L = 0; L < llm_config.n_layers; L++) {
                auto [k_cache, v_cache] = m_platform->ggml_backends[m_config->model_id]->m_kv->get_cache(L);
                bool is_need_bias = false;
                auto att_o = m_attn->build(g, x, L, g.add_tensor(k_cache), g.add_tensor(v_cache), batch_pos, mask, is_need_bias);
                auto ffn_o = m_ffn->build(g, att_o, L);
                x          = ffn_o;
            }

            // --- Last Batch Logic: Final Norm + Head + Softmax ---
            if (is_last_batch) {
                // 1. RMS Norm
                auto rms_final_w    = g.add_tensor(m_weights->rms_final_weight);
                auto final_rms_norm = g.rms_norm(x, rms_final_w, llm_config.norm_eps);
                
                // 2. Output Projection (to logits [batch, 2])
                auto output_w       = g.add_tensor(m_weights->output_weight);
                x                   = g.mat_mul(output_w, final_rms_norm);
                
                // 3. Softmax (In-Graph)
                x                   = g.softmax(x);
            }
        }

        // 4. Execution
        Executor executor(*m_platform, g);
        executor.allocate_buffers();
        POWERSERVE_LOG_INFO("Rerank: Executing batch of size {}", current_bs);
        executor.run();
        POWERSERVE_LOG_INFO("Rerank: Execution complete for batch size {}", current_bs);

#if defined(POWERSERVE_WITH_QNN)
        if (!m_platform->qnn_backend)
#endif
        {
            m_platform->ggml_backends[m_config->model_id]->m_kv->advance(current_bs);
        }

        // 5. Extract Result (Only if last batch)
        if (is_last_batch) {
            // x->m_data now contains probabilities [batch, 2]
            // We need the result for the very last token
            size_t valid_token_idx = current_bs - 1;
            // Assuming output dimension is 2 (Yes/No)
            // Safety: ensure dimension logic is sound. GGUF dump says 2.
            size_t dim = x->m_shape[0]; 
            if (dim != 2) {
                POWERSERVE_LOG_ERROR("Rerank output dimension mismatch, expected 2 but got {}", dim);
                POWERSERVE_ASSERT(false, "Qwen3-Reranker expect 2 for output dimension.");
            } // Qwen3-Reranker output dim is 2
            float* probs = static_cast<float*>(dynamic_cast<CPUBuffer&>(*x->m_data).m_data);
            float score = probs[valid_token_idx * dim + 0]; // Index 0 is 'yes'

            m_platform->ggml_backends[m_config->model_id]->reset_threadpool();

            return score;
        }

        n_processed += current_bs;
    }

    return 0.0f; // Should not reach here
}

} // namespace powerserve
