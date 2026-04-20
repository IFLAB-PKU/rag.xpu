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

#pragma once

/*
 * @ref: https://platform.openai.com/docs/api-reference
 */

#include "backend/platform.hpp"
#include "concurrentqueue.h"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "core/timer.hpp"
#include "core/typedefs.hpp"
#include "model/model.hpp"
#include "model/model_loader.hpp"
#include "model/module/norm_attention.hpp"
#include "sampler/sampler_chain.hpp"
#include "speculative/spec_model.hpp"

#include <cstddef>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

using ModelChatHistroyEntry = powerserve::ChatEntry;

struct ModelInput {

    /* Basic config */
    /// ID of the model to use.
    std::string m_model;

    /// [Only Completion] The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    std::string m_prompt;
    /// [Only Chat] The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens,
    /// or array of token arrays.
    std::vector<ModelChatHistroyEntry> m_history;

    /// The maximum number of tokens that can be generated in the completion.
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length.
    size_t m_max_num_token;

    /* Sample config */

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    float m_temperature;
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of
    /// the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are
    /// considered.
    float m_top_p;
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so
    /// far, increasing the model's likelihood to talk about new topics.
    float m_presence_penalty;
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text
    /// so far, decreasing the model's likelihood to repeat the same line verbatim.
    float m_frequency_penalty;

    /* Generation config */

    /// [Only Completion] How many completions to generate for each prompt.
    size_t m_response_n;
    /// [Only Completion] Generates best_of completions server-side and returns the "best" (the one with the highest log
    /// probability per token). Results cannot be streamed. When used with n, best_of controls the number of candidate
    /// completions and n specifies how many to return – best_of must be greater than n.
    size_t m_best_of_n;
    /// Include the log probabilities on the logprobs most likely output tokens, as well the chosen tokens. For example,
    /// if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob
    /// of the sampled token, so there may be up to logprobs+1 elements in the response. The maximum value for logprobs
    /// is 5.
    int m_log_probs;
    /// Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they
    /// become available, with the stream terminated by a data: [DONE] message.
    bool stream;

    /* Extension */

    float m_repeat_penalty;

    /* Rerank Extension */
    std::vector<std::string> m_documents; // lsh add for rerank
    size_t m_top_n;                       

    /* Metadata */

    size_t request_id;

    // unsupported: logit_bias
};

struct RerankResult {
    size_t index;
    float score;
};

struct ModelOutput {
    std::string m_text;
    std::vector<float> m_embedding; // lsh add for embedding
    std::vector<RerankResult> m_rerank_results; // lsh add for rerank
    size_t m_input_num_token;
    size_t m_output_num_token;
    std::optional<std::string> m_stop_reason;
};

// Contract object for segmented prefill -> decode handoff.
struct PrefillArtifact {
    size_t request_id = 0;
    std::string model_id;
    size_t prefill_tokens_total = 0;
    size_t kv_position_begin = 0;
    size_t kv_position_end = 0;
};

struct PrefillArtifactV2 {
    size_t request_id = 0;
    std::string model_id;
    std::string backend_id = "inproc";
    std::string session_id = "local";
    size_t segment_progress = 0;
    size_t prefill_tokens_total = 0;
    size_t kv_valid_begin = 0;
    size_t kv_valid_end = 0;
    std::string tokenizer_version = "default";
};

inline PrefillArtifactV2 make_prefill_artifact_v2(const PrefillArtifact &artifact, size_t segment_progress = 0) {
    return {
        .request_id = artifact.request_id,
        .model_id = artifact.model_id,
        .backend_id = "inproc",
        .session_id = "local",
        .segment_progress = segment_progress,
        .prefill_tokens_total = artifact.prefill_tokens_total,
        .kv_valid_begin = artifact.kv_position_begin,
        .kv_valid_end = artifact.kv_position_end,
        .tokenizer_version = "default",
    };
}

struct SegmentedPrefillQueueDemoOutput {
    ModelOutput output;
    PrefillArtifact prefill_artifact;
    size_t queue_wait_ms = 0;
};

struct GenerationDecodeTask {
    std::string query_type;
    size_t decode_rounds = 1;
    size_t decode_steps_per_round = 1;
    std::vector<std::string> input_segments;
};

struct GenerationDecodeCandidate {
    std::string source;
    ModelOutput output;
    size_t queue_wait_ms = 0;
    size_t rounds_executed = 0;
};

using ServerSessionId = int;

struct ServerSession {
    using ResultQueue = moodycamel::ConcurrentQueue<ModelOutput>;

    static constexpr int MAX_CHUNK_SIZE = 16;

    static constexpr int MAX_NUM_GENERATOR = 1;

    static constexpr int MAX_NUM_CONSUMER = 1;

public:
    ModelInput m_input;

    ResultQueue m_result_queue;

    std::unique_ptr<std::thread> m_session_thread_ptr;

public:
    ServerSession() = default;

    ServerSession(const ModelInput &input) :
        m_input(input),
        m_result_queue(MAX_CHUNK_SIZE, MAX_NUM_GENERATOR, MAX_NUM_CONSUMER) {}

    ~ServerSession() noexcept {
        if (m_session_thread_ptr) {
            m_session_thread_ptr->join();
        }
    }

    ServerSession(ServerSession &&other) noexcept = default;

    ServerSession &operator=(ServerSession &&other) noexcept = default;

public:
    void init(std::function<void()> &&thread_func) {
        if (m_session_thread_ptr) {
            POWERSERVE_LOG_ERROR("trying to init a session twice");
        } else {
            m_session_thread_ptr = std::make_unique<std::thread>(std::move(thread_func));
        }
    }

    std::optional<ModelOutput> fetch_result() {
        std::vector<ModelOutput> output_array(MAX_CHUNK_SIZE);
        const size_t actual_num = m_result_queue.try_dequeue_bulk(output_array.begin(), MAX_CHUNK_SIZE);
        output_array.resize(actual_num);

        // merge outputs
        if (actual_num == 0) {
            return std::nullopt;
        }

        ModelOutput output{
            .m_text             = {},
            .m_input_num_token  = output_array.back().m_input_num_token,
            .m_output_num_token = output_array.back().m_output_num_token,
            .m_stop_reason      = output_array.back().m_stop_reason
        };
        for (const ModelOutput &entry : output_array) {
            output.m_text += entry.m_text;
        }

        return output;
    }
};

struct ModelContext {
public:
    powerserve::Config m_config;
    std::shared_ptr<powerserve::Model> m_model_ptr;
    std::shared_ptr<powerserve::Model> m_draft_model_ptr;
    std::unique_ptr<powerserve::Tokenizer> m_tokenizer_ptr;
    powerserve::SpeculativeConfig speculative_config;

public:
    ModelContext() = default;

    ModelContext(
        const powerserve::Config &config,
        std::shared_ptr<powerserve::Model> &&model_ptr,
        std::unique_ptr<powerserve::Tokenizer> &&tokenizer_ptr
    ) :
        m_config(config),
        m_model_ptr(std::move(model_ptr)),
        m_tokenizer_ptr(std::move(tokenizer_ptr)) {}

    ModelContext(
        const powerserve::Config &config,
        std::shared_ptr<powerserve::Model> &&model_ptr,
        std::shared_ptr<powerserve::Model> &&draft_model_ptr,
        std::unique_ptr<powerserve::Tokenizer> &&tokenizer_ptr
    ) :
        m_config(config),
        m_model_ptr(std::move(model_ptr)),
        m_draft_model_ptr(std::move(draft_model_ptr)),
        m_tokenizer_ptr(std::move(tokenizer_ptr)) {}

    ~ModelContext() noexcept = default;

    ModelContext(const ModelContext &other) = delete;

    ModelContext(ModelContext &&other) noexcept = default;

    ModelContext &operator=(const ModelContext &other) = delete;

    ModelContext &operator=(ModelContext &&other) noexcept = default;
};

struct ServerContext {
private:
    std::filesystem::path m_work_folder;

    std::filesystem::path m_lib_folder;

    std::mutex m_lock;

    std::shared_ptr<powerserve::Platform> m_platform_ptr;

    std::unordered_map<std::string, std::shared_ptr<powerserve::Model>> m_model_map;

    std::unordered_map<std::string, ModelContext> m_context_slot_map;

    std::map<ServerSessionId, ServerSession> m_session_map;

public:
    ServerContext(powerserve::Path work_folder, powerserve::Path lib_folder) :
        m_work_folder(std::move(work_folder)),
        m_lib_folder(std::move(lib_folder)),
        m_platform_ptr(std::make_shared<powerserve::Platform>()) {
        if (!std::filesystem::exists(m_work_folder)) {
            POWERSERVE_LOG_WARN("model base folder does not exist: {}", m_work_folder);
        }
        if (!std::filesystem::is_directory(m_work_folder)) {
            POWERSERVE_LOG_WARN("model base folder is not directory: {}", m_work_folder);
        }

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->init_qnn_backend(m_lib_folder);
#endif // POWERSERVE_WITH_QNN
    }

    ~ServerContext() = default;

public:
    ModelContext &setup_model_for_blocking_pd(const ModelInput &input) {
        constexpr size_t PD_SLOT_COUNT = 2;
        const std::string slot_model_name =
            input.m_model + std::string(PD_SLOT_TAG) + std::to_string(input.request_id % PD_SLOT_COUNT);
        return setup_model(slot_model_name);
    }

    ModelContext &setup_model(const std::string &model_name) {
        const std::string resolved_model_name = strip_pd_slot_suffix(model_name);

        // Parse model name
        std::string_view main_model_name;
        std::string_view draft_model_name;
        {
            const auto iter = resolved_model_name.find('+');
            if (iter == std::string::npos) {
                main_model_name = std::string_view(resolved_model_name);
            } else {
                main_model_name =
                    std::string_view(resolved_model_name.cbegin(), resolved_model_name.cbegin() + iter);
                draft_model_name =
                    std::string_view(resolved_model_name.cbegin() + iter + 1, resolved_model_name.cend());
            }
        }
        POWERSERVE_LOG_INFO("main model: {}, draft model: {}", main_model_name, draft_model_name);

        std::lock_guard<std::mutex> lock_guard(m_lock);

        if (m_context_slot_map.contains(model_name)) {
            POWERSERVE_LOG_INFO("found cached model context: {}", model_name);
            return m_context_slot_map.at(model_name);
        }

        const powerserve::Path main_model_path  = model_name_to_path(main_model_name);
        const powerserve::Path draft_model_path = draft_model_name.empty() ? "" : model_name_to_path(draft_model_name);

        const powerserve::Config workspace_config(
            m_work_folder, powerserve::Path(m_work_folder) / powerserve::WORKSPACE_CONFIG_FILENAME
        );

        const std::string tokenizer_path                     = main_model_path / powerserve::MODEL_VOCAB_FILENAME;
        std::unique_ptr<powerserve::Tokenizer> tokenizer_ptr = std::make_unique<powerserve::Tokenizer>(tokenizer_path);
        POWERSERVE_LOG_INFO("after tokenizer init: {}", powerserve::perf_get_mem_result());

        std::shared_ptr<powerserve::Model> main_model =
            init_model(main_model_path, workspace_config.hyper_params, model_name + "#main");
        if (draft_model_path.empty()) {
            m_context_slot_map[model_name] =
                ModelContext(workspace_config, std::move(main_model), std::move(tokenizer_ptr));
        } else {
            std::shared_ptr<powerserve::Model> draft_model =
                init_model(draft_model_path, workspace_config.hyper_params, model_name + "#draft");
#if defined(POWERSERVE_WITH_QNN)
            // Speculative decoding still relies on QNN KV interface operations (copy/move/mask).
            auto *main_qnn_kv  = m_platform_ptr->qnn_backend->get_kv_interface(main_model->m_config->model_id);
            auto *draft_qnn_kv = m_platform_ptr->qnn_backend->get_kv_interface(draft_model->m_config->model_id);
            POWERSERVE_ASSERT(main_qnn_kv != nullptr);
            POWERSERVE_ASSERT(draft_qnn_kv != nullptr);
            main_model->kv_cache  = main_qnn_kv;
            draft_model->kv_cache = draft_qnn_kv;
#endif // POWERSERVE_WITH_QNN
            m_context_slot_map[model_name] =
                ModelContext(workspace_config, std::move(main_model), std::move(draft_model), std::move(tokenizer_ptr));
        }

        return m_context_slot_map.at(model_name);
    }

    void destroy_model(const std::string &model_name) {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        if (!m_model_map.contains(model_name)) {
            return;
        }

        // Unload model from backends
        const auto &model = m_model_map[model_name];
        m_platform_ptr->destroy_ggml_backend(model->m_config);

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->qnn_backend->unload_model(model->m_config);
#endif // POWERSERVE_WITH_QNN

        // Erase model and model context
        m_context_slot_map.erase(model_name);
        m_model_map.erase(model_name);
    }

    void destroy_all_models() {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        destroy_all_models_unsafe();
    }

    std::vector<std::string> list_models() const {
        if (!std::filesystem::exists(m_work_folder) && !std::filesystem::is_directory(m_work_folder)) {
            POWERSERVE_LOG_ERROR("model base folder does not exist: {}", m_work_folder);
            return {};
        }

        std::vector<std::string> model_list;
        for (const auto &entry : std::filesystem::directory_iterator(m_work_folder)) {
            const std::string dir_name = entry.path().filename();

            // TODO: no hardcoded string
            if (dir_name == "bin" || dir_name == "qnn_libs") {
                continue;
            }

            if (!entry.is_directory()) {
                POWERSERVE_LOG_ERROR("model folder is not directory: {}", m_work_folder);
                continue;
            }
            model_list.emplace_back(entry.path().filename());
        }

        return model_list;
    }

public:
    ServerSessionId setup_session(const ModelInput &input) {
        static int counter = 0;
        std::lock_guard<std::mutex> lock_guard(m_lock);
        while (true) {
            const int new_id = counter++;
            if (m_session_map.contains(new_id)) {
                continue;
            }
            m_session_map[new_id] = ServerSession(input);

            POWERSERVE_LOG_INFO("set up session: {}", new_id);
            return new_id;
        }
    }

    ServerSession &get_session(const ServerSessionId session_id) {
        std::lock_guard<std::mutex> lock_guard(m_lock);
        return m_session_map[session_id];
    }

    void destroy_session(const ServerSessionId session_id) {
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            if (!m_session_map.contains(session_id)) {
                POWERSERVE_LOG_WARN("cannot destroy session with session id: {}", session_id);
                return;
            }
            m_session_map.erase(session_id);
        }
        POWERSERVE_LOG_INFO("destroy session: {}", session_id);
    }

private:
    static constexpr const char *PD_SLOT_TAG = "@pdslot";

    static bool is_pd_slot_model_name(const std::string &model_name) {
        return model_name.find(PD_SLOT_TAG) != std::string::npos;
    }

    static std::string strip_pd_slot_suffix(const std::string &model_name) {
        const auto pos = model_name.find(PD_SLOT_TAG);
        if (pos == std::string::npos) {
            return model_name;
        }
        return model_name.substr(0, pos);
    }

    powerserve::Path model_name_to_path(const std::string_view model_name) const {
        const powerserve::Path inner_model_folder = m_work_folder / model_name;
        powerserve::Path model_folder;
        if (std::filesystem::exists(inner_model_folder) && std::filesystem::is_directory(inner_model_folder)) {
            model_folder = inner_model_folder;
        } else if (std::filesystem::exists(model_name) && std::filesystem::is_directory(model_name)) {
            model_folder = model_name;
        } else {
            POWERSERVE_LOG_ERROR("model folder does not exist: {}", model_name);
            throw std::invalid_argument("model folder does not exist");
        }
        POWERSERVE_LOG_INFO("found model folder: {}", model_folder);
        return model_folder;
    }

    std::shared_ptr<powerserve::Model> init_model(
        const powerserve::Path &model_path,
        const powerserve::HyperParams &hyper_params,
        const std::string &instance_key
    ) {
        if (m_model_map.contains(instance_key)) {
            return m_model_map.at(instance_key);
        }

        std::shared_ptr<powerserve::Model> model_ptr = powerserve::load_model(model_path);

        const auto slot_pos = instance_key.find(PD_SLOT_TAG);
        if (slot_pos != std::string::npos) {
            const auto slot_suffix = instance_key.substr(slot_pos);
            // Keep original ModelConfig object alive because model modules
            // store references to m_config->llm fields.
            model_ptr->m_config->model_id = model_ptr->m_config->model_id + slot_suffix;
            if (!model_ptr->m_config->vision.model_id.empty()) {
                model_ptr->m_config->vision.model_id = model_ptr->m_config->model_id;
            }
        }

        model_ptr->m_platform = m_platform_ptr;
        m_platform_ptr->init_ggml_backend(model_ptr->m_config, hyper_params);
        model_ptr->kv_cache = m_platform_ptr->ggml_backends[model_ptr->m_config->model_id]->m_kv->kv_cache.get();
        POWERSERVE_ASSERT(model_ptr->kv_cache != nullptr);

        model_ptr->m_attn = std::make_shared<powerserve::NormAttention>(model_ptr->m_config->llm, model_ptr->m_weights);
        POWERSERVE_LOG_INFO("after attn init: {}", powerserve::perf_get_mem_result());

#if defined(POWERSERVE_WITH_QNN)
        m_platform_ptr->qnn_backend->load_model(
            model_path / powerserve::qnn::QNN_WORKSPACE_DIR_NAME, model_ptr->m_config
        );
#endif // POWERSERVE_WITH_QNN

        m_model_map[instance_key] = model_ptr;
        return model_ptr;
    }

    void destroy_all_models_unsafe() {
        POWERSERVE_LOG_INFO("Destory all models in memory");
        for (const auto &[_, model] : m_model_map) {
            m_platform_ptr->destroy_ggml_backend(model->m_config);
#if defined(POWERSERVE_WITH_QNN)
            m_platform_ptr->qnn_backend->unload_model(model->m_config);
#endif // POWERSERVE_WITH_QNN
        }

        m_context_slot_map.clear();
        m_model_map.clear();
    }
};

/*!
 * @param output_string[in] The output string after tokenization of model
 * @note For some reasons(e.g. truncation), the output token may be incomplete. In case of json parser exception,
 * we need to hold the incomplete word until next time or the end.
 */
inline bool is_utf8_string_incomplete(const std::string &output_string) {
    bool incomplete = false;
    for (unsigned i = 1; i < 5 && i <= output_string.size(); ++i) {
        unsigned char c = output_string[output_string.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }
    return incomplete;
}

inline std::string &remove_incomplete_utf8_char(std::string &output_string) {
    for (unsigned i = 1; i < 5 && i <= output_string.size(); ++i) {
        unsigned char c = output_string[output_string.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            if (i < 2) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            if (i < 3) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            if (i < 4) {
                output_string.erase(output_string.size() - i, i);
                return output_string;
            };
        }
        // else 1-byte character or invalid byte
        break;
    }
    POWERSERVE_LOG_INFO("The output string is completed");
    return output_string;
}

inline void stream_inference(const ModelContext &context, ServerSession &session, const std::string &input_prompt) {
    using namespace powerserve;

    const ModelInput &input = session.m_input;

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &draft     = context.m_draft_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    // TODO: This sampler config is too argly
    HyperParams::SamplerConfig sampler_config = config.hyper_params.sampler_config;
    sampler_config.temperature                = input.m_temperature;
    sampler_config.penalty_freq               = input.m_frequency_penalty;
    sampler_config.penalty_present            = input.m_presence_penalty;
    sampler_config.penalty_repeat             = input.m_repeat_penalty;
    sampler_config.top_p                      = input.m_top_p;
    sampler_config.temperature                = input.m_temperature;
    powerserve::SamplerChain sampler{sampler_config, tokenizer};

    /* Inference */
    ModelOutput output;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    std::string stop_reason = "length";
    size_t step             = 0;

    POWERSERVE_LOG_DEBUG("Model input     : {}", powerserve::abbreviation(input_prompt, 50));
    POWERSERVE_LOG_DEBUG("Model max token : {}", max_num_token);
    POWERSERVE_LOG_DEBUG("Model batch size: {}", batch_size);

    /*
     * Prefill
     */
    Timer timer;
    Timer total_timer;
    const size_t num_prefill_token = tokenizer.tokenize(input_prompt, tokenizer.m_vocab.tokenizer_add_bos).size() - 1;
    bool first_chunk_emitted       = false;
    double ttft_ms                 = -1.0;

    bool end_of_text = false;
    std::string output_buffer;

    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (draft) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(
            context.m_model_ptr, context.m_draft_model_ptr, context.speculative_config
        );
        iter = spec_model->generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    } else
#endif
    {
        iter = model.generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    }
    const size_t prefill_time_ms = timer.elapsed_time_ms();

    while (!iter->end()) {
        auto token = iter->next();
        step++;
        if (step == 1) {
            POWERSERVE_LOG_INFO(
                "prefill step: {}, prefill time: {}ms ({} token/s)",
                num_prefill_token,
                prefill_time_ms,
                num_prefill_token * 1000.f / prefill_time_ms
            );
            timer.reset();
            continue;
        } // Avoid outputting the last token

        if (token == tokenizer.bos_token()) {
            continue;
        }

        if (tokenizer.should_stop(token)) {
            end_of_text = true;
            break;
        } else {
            output_buffer += tokenizer.to_string(token);
            if (!is_utf8_string_incomplete(output_buffer)) {
                if (!first_chunk_emitted) {
                    first_chunk_emitted = true;
                    ttft_ms             = static_cast<double>(total_timer.elapsed_time_ms());
                }
                session.m_result_queue.enqueue(
                    {.m_text             = output_buffer,
                     .m_input_num_token  = num_prefill_token,
                     .m_output_num_token = step,
                     .m_stop_reason      = std::nullopt}
                );
                output_buffer.clear();
            }
        }
    }

    const std::string_view end_text = end_of_text ? "[end of text]" : "";

    session.m_result_queue.enqueue(
        {.m_text             = remove_incomplete_utf8_char(output_buffer).append(end_text),
         .m_input_num_token  = num_prefill_token,
         .m_output_num_token = step,
         .m_stop_reason      = end_of_text ? "stop" : "length"}
    );

    const size_t decode_time_ms = timer.elapsed_time_ms();
    const size_t total_time_ms  = total_timer.elapsed_time_ms();
    const size_t decode_steps   = step > 1 ? step - 1 : 0;
    const double prefill_tps    = prefill_time_ms > 0 ? num_prefill_token * 1000.0 / prefill_time_ms : 0.0;
    const double decode_tps     = decode_time_ms > 0 ? decode_steps * 1000.0 / decode_time_ms : 0.0;

    POWERSERVE_LOG_INFO(
        "decode  step: {}, decode  time: {}ms ({} token/s)", step, decode_time_ms, step * 1000.f / decode_time_ms
    );
    POWERSERVE_LOG_INFO(
        "PD_METRIC request_id={},prefill_tokens={},decode_tokens={},T_total_ms={},T_prefill_ms={},T_decode_ms={},TTFT_ms={},prefill_tps={},decode_tps={}",
        input.request_id,
        num_prefill_token,
        decode_steps,
        total_time_ms,
        prefill_time_ms,
        decode_time_ms,
        ttft_ms,
        prefill_tps,
        decode_tps
    );
}

struct BlockingPrefillResult {
    std::shared_ptr<powerserve::TokenIterator> iter;
    size_t num_prefill_token;
    size_t prefill_time_ms;
};

inline PrefillArtifact build_prefill_artifact(
    const ModelContext &context,
    const ModelInput &input,
    BlockingPrefillResult &prefill_result,
    size_t kv_position_begin
) {
    POWERSERVE_ASSERT(context.m_model_ptr != nullptr);
    POWERSERVE_ASSERT(context.m_model_ptr->m_platform != nullptr);

    std::string model_id = context.m_model_ptr->m_config->model_id;
    const size_t kv_position_end = context.m_model_ptr->m_platform->get_kv_position(model_id);

    return {
        .request_id = input.request_id,
        .model_id = model_id,
        .prefill_tokens_total = prefill_result.num_prefill_token,
        .kv_position_begin = kv_position_begin,
        .kv_position_end = kv_position_end,
    };
}

inline BlockingPrefillResult run_blocking_prefill(
    const ModelContext &context, const ModelInput &input, const std::string &input_prompt, powerserve::SamplerChain &sampler
) {
    using namespace powerserve;

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &draft     = context.m_draft_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    bool add_special_tokens         = tokenizer.m_vocab.tokenizer_add_bos || tokenizer.m_vocab.tokenizer_add_eos;
    const size_t num_prefill_token  = tokenizer.tokenize(input_prompt, add_special_tokens).size() - 1;

    Timer prefill_timer;
    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;
#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (draft) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(
            context.m_model_ptr, context.m_draft_model_ptr, context.speculative_config
        );
        iter = spec_model->generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    } else
#endif
    {
        iter = model.generate(tokenizer, sampler, input_prompt, max_num_token, batch_size);
    }

    return {
        .iter              = std::move(iter),
        .num_prefill_token = num_prefill_token,
        .prefill_time_ms   = static_cast<size_t>(prefill_timer.elapsed_time_ms())
    };
}

inline BlockingPrefillResult run_blocking_prefill_segmented(
    const ModelContext &context,
    const ModelInput &input,
    const std::vector<std::string> &input_segments,
    powerserve::SamplerChain &sampler
) {
    using namespace powerserve;

    if (input_segments.empty()) {
        return run_blocking_prefill(context, input, input.m_prompt, sampler);
    }

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &draft     = context.m_draft_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    bool add_special_tokens = tokenizer.m_vocab.tokenizer_add_bos || tokenizer.m_vocab.tokenizer_add_eos;

    Timer prefill_timer;
    std::shared_ptr<powerserve::TokenIterator> iter = nullptr;

    const std::string &first_segment = input_segments.front();
    const std::vector<powerserve::Token> first_segment_tokens = tokenizer.tokenize(first_segment, add_special_tokens);
    size_t num_prefill_token = first_segment_tokens.empty() ? 0 : first_segment_tokens.size() - 1;

#if defined(POWERSERVE_WITH_QNN)
    std::shared_ptr<powerserve::SpeculativeModel> spec_model = nullptr;
    if (draft) {
        spec_model = std::make_shared<powerserve::SpeculativeModel>(
            context.m_model_ptr, context.m_draft_model_ptr, context.speculative_config
        );
        iter = spec_model->generate(tokenizer, sampler, first_segment, max_num_token, batch_size);
    } else
#endif
    {
        iter = model.generate(tokenizer, sampler, first_segment, max_num_token, batch_size);
    }

    auto segmented_iter = std::dynamic_pointer_cast<powerserve::ModelTokenIterator>(iter);
    if (segmented_iter == nullptr) {
        // Fallback for unsupported iterator implementations.
        std::string merged_prompt;
        for (const auto &segment : input_segments) {
            merged_prompt += segment;
        }
        return run_blocking_prefill(context, input, merged_prompt, sampler);
    }

    for (size_t i = 1; i < input_segments.size(); ++i) {
        const std::vector<powerserve::Token> segment_tokens = tokenizer.tokenize(input_segments[i], false);
        if (segment_tokens.empty()) {
            continue;
        }
        num_prefill_token += segment_tokens.size();
        segmented_iter->prefill_segment_tokens(segment_tokens);
        POWERSERVE_LOG_DEBUG(
            "segmented prefill request_id={}, segment={}, tokens={}", input.request_id, i + 1, segment_tokens.size()
        );
    }

    return {
        .iter              = std::move(iter),
        .num_prefill_token = num_prefill_token,
        .prefill_time_ms   = static_cast<size_t>(prefill_timer.elapsed_time_ms())
    };
}

inline ModelOutput run_blocking_decode_steps_from_artifact(
    const ModelInput &input,
    const powerserve::Tokenizer &tokenizer,
    BlockingPrefillResult &prefill_result,
    size_t max_decode_steps,
    bool has_prefill_boundary = true
) {
    using namespace powerserve;

    std::string output_text;
    std::string stop_reason = "length";
    size_t step             = 0;
    size_t decode_steps     = 0;
    bool end_of_text        = false;
    double ttft_ms          = -1.0;

    Timer decode_timer;

    while (!prefill_result.iter->end()) {
        auto token = prefill_result.iter->next();
        step++;

        if (has_prefill_boundary && step == 1) {
            const double prefill_tps_log = prefill_result.prefill_time_ms > 0
                ? prefill_result.num_prefill_token * 1000.0 / prefill_result.prefill_time_ms
                : 0.0;
            POWERSERVE_LOG_INFO(
                "prefill step: {}, prefill time: {}ms ({} token/s)",
                prefill_result.num_prefill_token,
                prefill_result.prefill_time_ms,
                prefill_tps_log
            );
            decode_timer.reset();
            continue;
        } // Avoid outputting the last token

        if (decode_steps >= max_decode_steps) {
            break;
        }
        decode_steps++;

        if (token == tokenizer.bos_token()) {
            continue;
        }

        if (tokenizer.should_stop(token)) {
            end_of_text = true;
            stop_reason = "stop";
            break;
        } else {
            const std::string token_text = tokenizer.to_string(token);
            if (ttft_ms < 0.0 && !token_text.empty()) {
                ttft_ms = static_cast<double>(prefill_result.prefill_time_ms + decode_timer.elapsed_time_ms());
            }
            output_text += token_text;
        }
    }

    remove_incomplete_utf8_char(output_text);
    output_text += end_of_text ? "[end of text]" : "";

    const size_t decode_time_ms = decode_timer.elapsed_time_ms();
    const size_t total_time_ms  = prefill_result.prefill_time_ms + decode_time_ms;
    const double prefill_tps    = prefill_result.prefill_time_ms > 0
        ? prefill_result.num_prefill_token * 1000.0 / prefill_result.prefill_time_ms
        : 0.0;
    const double decode_tps = decode_time_ms > 0 ? decode_steps * 1000.0 / decode_time_ms : 0.0;

    POWERSERVE_LOG_INFO(
        "decode  step: {}, decode  time: {}ms ({} token/s)", step, decode_time_ms, step * 1000.f / decode_time_ms
    );
    POWERSERVE_LOG_INFO(
        "PD_METRIC request_id={},prefill_tokens={},decode_tokens={},T_total_ms={},T_prefill_ms={},T_decode_ms={},TTFT_ms={},prefill_tps={},decode_tps={}",
        input.request_id,
        prefill_result.num_prefill_token,
        decode_steps,
        total_time_ms,
        prefill_result.prefill_time_ms,
        decode_time_ms,
        ttft_ms,
        prefill_tps,
        decode_tps
    );

    return {
        .m_text             = output_text,
        .m_input_num_token  = prefill_result.num_prefill_token,
        .m_output_num_token = step,
        .m_stop_reason      = stop_reason
    };
}

inline ModelOutput run_blocking_decode_from_artifact(
    const ModelInput &input, const powerserve::Tokenizer &tokenizer, BlockingPrefillResult &prefill_result
) {
    return run_blocking_decode_steps_from_artifact(
        input,
        tokenizer,
        prefill_result,
        std::numeric_limits<size_t>::max()
    );
}

class PrefillExecutor {
public:
    virtual ~PrefillExecutor() = default;

    virtual BlockingPrefillResult run_prefill(
        const ModelContext &context,
        const ModelInput &input,
        const std::string &input_prompt,
        powerserve::SamplerChain &sampler
    ) = 0;

    virtual BlockingPrefillResult run_prefill_segmented(
        const ModelContext &context,
        const ModelInput &input,
        const std::vector<std::string> &input_segments,
        powerserve::SamplerChain &sampler
    ) = 0;
};

class DecodeExecutor {
public:
    virtual ~DecodeExecutor() = default;

    virtual ModelOutput run_decode_steps(
        const ModelInput &input,
        const powerserve::Tokenizer &tokenizer,
        BlockingPrefillResult &prefill_result,
        size_t max_decode_steps,
        bool has_prefill_boundary = true
    ) = 0;

    virtual ModelOutput run_decode(
        const ModelInput &input,
        const powerserve::Tokenizer &tokenizer,
        BlockingPrefillResult &prefill_result
    ) = 0;
};

class LocalPrefillExecutor final : public PrefillExecutor {
public:
    BlockingPrefillResult run_prefill(
        const ModelContext &context,
        const ModelInput &input,
        const std::string &input_prompt,
        powerserve::SamplerChain &sampler
    ) override {
        return run_blocking_prefill(context, input, input_prompt, sampler);
    }

    BlockingPrefillResult run_prefill_segmented(
        const ModelContext &context,
        const ModelInput &input,
        const std::vector<std::string> &input_segments,
        powerserve::SamplerChain &sampler
    ) override {
        return run_blocking_prefill_segmented(context, input, input_segments, sampler);
    }
};

class LocalDecodeExecutor final : public DecodeExecutor {
public:
    ModelOutput run_decode_steps(
        const ModelInput &input,
        const powerserve::Tokenizer &tokenizer,
        BlockingPrefillResult &prefill_result,
        size_t max_decode_steps,
        bool has_prefill_boundary = true
    ) override {
        return run_blocking_decode_steps_from_artifact(
            input,
            tokenizer,
            prefill_result,
            max_decode_steps,
            has_prefill_boundary
        );
    }

    ModelOutput run_decode(
        const ModelInput &input,
        const powerserve::Tokenizer &tokenizer,
        BlockingPrefillResult &prefill_result
    ) override {
        return run_blocking_decode_from_artifact(input, tokenizer, prefill_result);
    }
};

inline GenerationDecodeCandidate run_blocking_decode_rounds_from_artifact(
    DecodeExecutor &decode_executor,
    const ModelInput &input,
    const powerserve::Tokenizer &tokenizer,
    BlockingPrefillResult &prefill_result,
    const GenerationDecodeTask &decode_task
) {
    const size_t max_rounds = decode_task.decode_rounds == 0 ? 1 : decode_task.decode_rounds;
    const size_t steps_per_round = decode_task.decode_steps_per_round == 0 ? 1 : decode_task.decode_steps_per_round;

    ModelOutput merged_output{
        .m_text = {},
        .m_embedding = {},
        .m_rerank_results = {},
        .m_input_num_token = prefill_result.num_prefill_token,
        .m_output_num_token = 1,
        .m_stop_reason = std::string{"length"}
    };

    size_t total_decode_tokens = 0;
    size_t rounds_executed = 0;
    for (size_t round = 0; round < max_rounds; ++round) {
        if (prefill_result.iter->end()) {
            break;
        }

        ModelOutput round_output = decode_executor.run_decode_steps(
            input,
            tokenizer,
            prefill_result,
            steps_per_round,
            round == 0
        );
        ++rounds_executed;

        if (!round_output.m_text.empty()) {
            merged_output.m_text += round_output.m_text;
        }

        size_t round_decode_tokens = round_output.m_output_num_token;
        if (round == 0 && round_decode_tokens > 0) {
            round_decode_tokens -= 1;
        }
        total_decode_tokens += round_decode_tokens;

        merged_output.m_stop_reason = round_output.m_stop_reason;
        if (round_output.m_stop_reason.has_value() && round_output.m_stop_reason.value() == "stop") {
            break;
        }
    }

    merged_output.m_output_num_token = total_decode_tokens + 1;

    return {
        .source = decode_task.query_type,
        .output = std::move(merged_output),
        .queue_wait_ms = 0,
        .rounds_executed = rounds_executed,
    };
}

inline ModelOutput blocking_inference_segmented_prefill_prototype(
    const ModelContext &context,
    const ModelInput &input,
    const std::vector<std::string> &input_segments,
    size_t max_decode_steps = 1
) {
    using namespace powerserve;

    const auto &tokenizer = *context.m_tokenizer_ptr;

    auto sampler_config           = context.m_config.hyper_params.sampler_config;
    sampler_config.temperature    = input.m_temperature;
    sampler_config.penalty_freq   = input.m_frequency_penalty;
    sampler_config.penalty_present = input.m_presence_penalty;
    sampler_config.penalty_repeat = input.m_repeat_penalty;
    sampler_config.top_p          = input.m_top_p;
    sampler_config.temperature    = input.m_temperature;
    powerserve::SamplerChain sampler{sampler_config, tokenizer};
    LocalPrefillExecutor prefill_executor;
    LocalDecodeExecutor decode_executor;

    BlockingPrefillResult prefill_result = prefill_executor.run_prefill_segmented(
        context,
        input,
        input_segments,
        sampler
    );

    return decode_executor.run_decode_steps(input, tokenizer, prefill_result, max_decode_steps);
}

inline std::shared_ptr<std::mutex> get_or_create_model_exec_mutex(const std::string &model_id);
inline std::string get_model_exec_lock_key(const ModelContext &context);
inline std::unique_lock<std::mutex> lock_model_execution(const ModelContext &context);

class SequentialSegmentPrefillQueueDemo {
public:
    SequentialSegmentPrefillQueueDemo() {
        m_worker = std::thread([this]() { worker_loop(); });
    }

    ~SequentialSegmentPrefillQueueDemo() {
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            m_shutdown = true;
        }
        m_cv.notify_all();
        if (m_worker.joinable()) {
            m_worker.join();
        }
    }

    SegmentedPrefillQueueDemoOutput run(
        const ModelContext &context,
        const ModelInput &input,
        const std::vector<std::string> &input_segments,
        size_t max_decode_steps
    ) {
        SegmentedPrefillQueueTask task;
        task.context = &context;
        task.input = input;
        task.input_segments = input_segments;
        task.max_decode_steps = max_decode_steps;
        task.enqueued_at = std::chrono::steady_clock::now();

        auto result_future = task.result_promise.get_future();
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            m_queue.push_back(std::move(task));
        }
        m_cv.notify_one();
        return result_future.get();
    }

    GenerationDecodeCandidate run_multiround(
        const ModelContext &context,
        const ModelInput &input,
        const GenerationDecodeTask &decode_task
    ) {
        SegmentedPrefillQueueTask task;
        task.context = &context;
        task.input = input;
        task.input_segments = decode_task.input_segments;
        task.max_decode_steps = decode_task.decode_steps_per_round;
        task.multiround_mode = true;
        task.decode_task = decode_task;
        task.enqueued_at = std::chrono::steady_clock::now();

        auto result_future = task.multiround_result_promise.get_future();
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            m_queue.push_back(std::move(task));
        }
        m_cv.notify_one();
        return result_future.get();
    }

private:
    struct SegmentedPrefillQueueTask {
        const ModelContext *context = nullptr;
        ModelInput input;
        std::vector<std::string> input_segments;
        size_t max_decode_steps = 1;
        bool multiround_mode = false;
        GenerationDecodeTask decode_task;
        std::chrono::steady_clock::time_point enqueued_at;
        std::promise<SegmentedPrefillQueueDemoOutput> result_promise;
        std::promise<GenerationDecodeCandidate> multiround_result_promise;
    };

    std::deque<SegmentedPrefillQueueTask> m_queue;
    std::mutex m_lock;
    std::condition_variable m_cv;
    std::thread m_worker;
    bool m_shutdown = false;
    LocalPrefillExecutor m_prefill_executor;
    LocalDecodeExecutor m_decode_executor;

    void worker_loop() {
        while (true) {
            SegmentedPrefillQueueTask task;
            {
                std::unique_lock<std::mutex> lock(m_lock);
                m_cv.wait(lock, [this]() { return m_shutdown || !m_queue.empty(); });
                if (m_shutdown && m_queue.empty()) {
                    return;
                }
                task = std::move(m_queue.front());
                m_queue.pop_front();
            }

            try {
                POWERSERVE_ASSERT(task.context != nullptr);
                const auto &context   = *task.context;
                const auto &tokenizer = *context.m_tokenizer_ptr;
                const auto &input = task.input;

                const auto now = std::chrono::steady_clock::now();
                const size_t queue_wait_ms = static_cast<size_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - task.enqueued_at).count()
                );

                auto model_exec_lock = lock_model_execution(context);

                auto sampler_config           = context.m_config.hyper_params.sampler_config;
                sampler_config.temperature    = input.m_temperature;
                sampler_config.penalty_freq   = input.m_frequency_penalty;
                sampler_config.penalty_present = input.m_presence_penalty;
                sampler_config.penalty_repeat = input.m_repeat_penalty;
                sampler_config.top_p          = input.m_top_p;
                sampler_config.temperature    = input.m_temperature;
                powerserve::SamplerChain sampler{sampler_config, tokenizer};

                std::string model_id = context.m_model_ptr->m_config->model_id;
                const size_t kv_position_begin = context.m_model_ptr->m_platform->get_kv_position(model_id);
                BlockingPrefillResult prefill_result = m_prefill_executor.run_prefill_segmented(
                    context,
                    input,
                    task.input_segments,
                    sampler
                );
                PrefillArtifact prefill_artifact = build_prefill_artifact(
                    context,
                    input,
                    prefill_result,
                    kv_position_begin
                );

                if (task.multiround_mode) {
                    GenerationDecodeCandidate candidate = run_blocking_decode_rounds_from_artifact(
                        m_decode_executor,
                        input,
                        tokenizer,
                        prefill_result,
                        task.decode_task
                    );
                    candidate.queue_wait_ms = queue_wait_ms;
                    task.multiround_result_promise.set_value(std::move(candidate));
                } else {
                    ModelOutput output = m_decode_executor.run_decode_steps(
                        input,
                        tokenizer,
                        prefill_result,
                        task.max_decode_steps
                    );
                    task.result_promise.set_value({
                        .output = std::move(output),
                        .prefill_artifact = std::move(prefill_artifact),
                        .queue_wait_ms = queue_wait_ms,
                    });
                }
            } catch (...) {
                if (task.multiround_mode) {
                    task.multiround_result_promise.set_exception(std::current_exception());
                } else {
                    task.result_promise.set_exception(std::current_exception());
                }
            }
        }
    }
};

inline SegmentedPrefillQueueDemoOutput blocking_inference_segmented_prefill_queue_demo(
    const ModelContext &context,
    const ModelInput &input,
    const std::vector<std::string> &input_segments,
    size_t max_decode_steps = 1
) {
    static SequentialSegmentPrefillQueueDemo queue_demo;
    return queue_demo.run(context, input, input_segments, max_decode_steps);
}

inline GenerationDecodeCandidate blocking_inference_segmented_prefill_multiround_decode(
    const ModelContext &context,
    const ModelInput &input,
    const GenerationDecodeTask &decode_task
) {
    static SequentialSegmentPrefillQueueDemo queue_demo;
    return queue_demo.run_multiround(context, input, decode_task);
}

struct PDOrchestratorTask {
    const ModelContext *context = nullptr;
    const ModelInput *input     = nullptr;
    std::string input_prompt;
    std::unique_ptr<powerserve::SamplerChain> sampler_ptr;
    std::optional<BlockingPrefillResult> prefill_result;
    std::shared_ptr<std::mutex> model_exec_mutex;
    std::unique_lock<std::mutex> model_exec_lock;
    std::promise<ModelOutput> result_promise;
};

inline std::shared_ptr<std::mutex> get_or_create_model_exec_mutex(const std::string &model_id) {
    static std::mutex registry_lock;
    static std::unordered_map<std::string, std::shared_ptr<std::mutex>> model_exec_lock_map;

    std::lock_guard<std::mutex> lock_guard(registry_lock);
    auto iter = model_exec_lock_map.find(model_id);
    if (iter != model_exec_lock_map.end()) {
        return iter->second;
    }

    auto mutex_ptr = std::make_shared<std::mutex>();
    model_exec_lock_map.emplace(model_id, mutex_ptr);
    return mutex_ptr;
}

inline std::string get_model_exec_lock_key(const ModelContext &context) {
    POWERSERVE_ASSERT(context.m_model_ptr != nullptr);
    return context.m_model_ptr->m_config->model_id;
}

inline std::unique_lock<std::mutex> lock_model_execution(const ModelContext &context) {
    const std::string lock_key = get_model_exec_lock_key(context);
    const auto mutex_ptr = get_or_create_model_exec_mutex(lock_key);
    POWERSERVE_ASSERT(mutex_ptr != nullptr);
    return std::unique_lock<std::mutex>(*mutex_ptr);
}

class PDOrchestrator {
public:
    PDOrchestrator() {
        m_prefill_worker = std::thread([this]() { prefill_worker_loop(); });
        m_decode_worker  = std::thread([this]() { decode_worker_loop(); });
    }

    ~PDOrchestrator() {
        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            m_shutdown = true;
        }
        m_prefill_cv.notify_all();
        m_decode_cv.notify_all();

        if (m_prefill_worker.joinable()) {
            m_prefill_worker.join();
        }
        if (m_decode_worker.joinable()) {
            m_decode_worker.join();
        }
    }

    ModelOutput run_single_prompt(const ModelContext &context, const ModelInput &input, const std::string &input_prompt) {
        PDOrchestratorTask task{
            .context      = &context,
            .input        = &input,
            .input_prompt = input_prompt,
        };
        std::future<ModelOutput> result_future = task.result_promise.get_future();

        {
            std::lock_guard<std::mutex> lock_guard(m_lock);
            m_prefill_queue.push_back(std::move(task));
        }
        POWERSERVE_LOG_INFO("pd orchestrator: request {} queued to prefill", input.request_id);
        m_prefill_cv.notify_one();

        return result_future.get();
    }

private:
    std::deque<PDOrchestratorTask> m_prefill_queue;
    std::deque<PDOrchestratorTask> m_decode_queue;
    std::mutex m_lock;
    std::condition_variable m_prefill_cv;
    std::condition_variable m_decode_cv;
    std::thread m_prefill_worker;
    std::thread m_decode_worker;
    bool m_shutdown = false;
    LocalPrefillExecutor m_prefill_executor;
    LocalDecodeExecutor m_decode_executor;

    void prefill_worker_loop() {
        while (true) {
            PDOrchestratorTask task;
            {
                std::unique_lock<std::mutex> lock(m_lock);
                m_prefill_cv.wait(lock, [this]() { return m_shutdown || !m_prefill_queue.empty(); });
                if (m_shutdown && m_prefill_queue.empty()) {
                    return;
                }
                task = std::move(m_prefill_queue.front());
                m_prefill_queue.pop_front();
            }

            try {
                POWERSERVE_ASSERT(task.context != nullptr);
                POWERSERVE_ASSERT(task.input != nullptr);

                const auto &context   = *task.context;
                const auto &input     = *task.input;
                const auto &tokenizer = *context.m_tokenizer_ptr;

                auto sampler_config       = context.m_config.hyper_params.sampler_config;
                sampler_config.temperature     = input.m_temperature;
                sampler_config.penalty_freq    = input.m_frequency_penalty;
                sampler_config.penalty_present = input.m_presence_penalty;
                sampler_config.penalty_repeat  = input.m_repeat_penalty;
                sampler_config.top_p           = input.m_top_p;
                sampler_config.temperature     = input.m_temperature;
                task.sampler_ptr = std::make_unique<powerserve::SamplerChain>(sampler_config, tokenizer);

                task.model_exec_mutex = get_or_create_model_exec_mutex(get_model_exec_lock_key(context));
                POWERSERVE_ASSERT(task.model_exec_mutex != nullptr);
                task.model_exec_lock = std::unique_lock<std::mutex>(*task.model_exec_mutex);

                POWERSERVE_LOG_INFO("pd orchestrator: prefill request {} start", input.request_id);
                task.prefill_result = m_prefill_executor.run_prefill(
                    context,
                    input,
                    task.input_prompt,
                    *task.sampler_ptr
                );

                {
                    std::lock_guard<std::mutex> lock_guard(m_lock);
                    m_decode_queue.push_back(std::move(task));
                }
                POWERSERVE_LOG_INFO("pd orchestrator: request {} moved to decode", input.request_id);
                m_decode_cv.notify_one();
            } catch (...) {
                task.result_promise.set_exception(std::current_exception());
            }
        }
    }

    void decode_worker_loop() {
        while (true) {
            PDOrchestratorTask task;
            {
                std::unique_lock<std::mutex> lock(m_lock);
                m_decode_cv.wait(lock, [this]() { return m_shutdown || !m_decode_queue.empty(); });
                if (m_shutdown && m_decode_queue.empty()) {
                    return;
                }
                task = std::move(m_decode_queue.front());
                m_decode_queue.pop_front();
            }

            try {
                POWERSERVE_ASSERT(task.context != nullptr);
                POWERSERVE_ASSERT(task.input != nullptr);
                POWERSERVE_ASSERT(task.sampler_ptr != nullptr);
                POWERSERVE_ASSERT(task.prefill_result.has_value());

                const auto &context   = *task.context;
                const auto &input     = *task.input;
                const auto &tokenizer = *context.m_tokenizer_ptr;

                POWERSERVE_LOG_INFO("pd orchestrator: decode request {} start", input.request_id);
                ModelOutput output = m_decode_executor.run_decode(
                    input,
                    tokenizer,
                    task.prefill_result.value()
                );
                task.result_promise.set_value(std::move(output));
            } catch (...) {
                task.result_promise.set_exception(std::current_exception());
            }
        }
    }
};

inline ModelOutput blocking_inference(
    const ModelContext &context, const ModelInput &input, const std::string &input_prompt
) {
    using namespace powerserve;

    auto &config    = context.m_config;

    const size_t max_num_token = input.m_max_num_token;
    const size_t batch_size    = config.hyper_params.batch_size;

    POWERSERVE_LOG_DEBUG("Model input     : {}", powerserve::abbreviation(input_prompt, 20));
    POWERSERVE_LOG_DEBUG("Model max token : {}", max_num_token);
    POWERSERVE_LOG_DEBUG("Model batch size: {}", batch_size);

    static PDOrchestrator orchestrator;
    return orchestrator.run_single_prompt(context, input, input_prompt);
}

inline ModelOutput blocking_embedding(
    const ModelContext &context, const ModelInput &input, const std::string &input_prompt
) {
    using namespace powerserve;

    auto &config    = context.m_config;
    auto &model     = *context.m_model_ptr;
    auto &tokenizer = *context.m_tokenizer_ptr;

    ModelOutput output;
    const size_t batch_size = config.hyper_params.batch_size;

    POWERSERVE_LOG_DEBUG("Model input     : {}", powerserve::abbreviation(input_prompt, 20));
    POWERSERVE_LOG_DEBUG("Model batch size: {}", batch_size);

    /*
     * Embedding
     */
    Timer timer;
    // [FIX] Use `tokenizer.m_vocab.tokenizer_add_eos` as 2nd param for `tokenize`
    bool add_special_tokens = tokenizer.m_vocab.tokenizer_add_bos || tokenizer.m_vocab.tokenizer_add_eos;
    std::vector<powerserve::Token> tokens = tokenizer.tokenize(input_prompt, add_special_tokens);
    POWERSERVE_LOG_DEBUG("Tokenized input :{}", tokens);
    const size_t num_tokens = tokens.size();

    std::vector<float> embedding_vector;
    embedding_vector = model.compute_embedding(tokens, batch_size);

    const size_t latency_ms = timer.elapsed_time_ms();
    POWERSERVE_LOG_INFO(
        "embedding tokens: {}, embedding time: {}ms",
        num_tokens,
        latency_ms
    );

    return {
        .m_text             = "",
        .m_embedding        = embedding_vector,
        .m_input_num_token  = num_tokens,
        .m_output_num_token = 0,
        .m_stop_reason      = "stop"
    };


}

inline ModelOutput blocking_rerank(
    const ModelContext &context, const ModelInput &input
) {
    using namespace powerserve;

    auto &config      = context.m_config;
    auto &model       = *context.m_model_ptr;
    auto &tokenizer   = *context.m_tokenizer_ptr;

    ModelOutput output;
    const size_t batch_size = config.hyper_params.batch_size;
    output.m_input_num_token = 0;
    
    const std::string& query = input.m_prompt;
    const std::vector<std::string>& documents = input.m_documents;
    
    std::vector<RerankResult> all_results;
    all_results.reserve(documents.size());

    POWERSERVE_LOG_INFO("Rerank task: query='{}', docs_count={}", abbreviation(query, 20), documents.size());
    Timer timer;
    
    
    for (size_t i = 0; i < documents.size(); ++i) {
        const std::string& doc = documents[i];

        // 1. Make Prompt according to template
        std::string prompt = tokenizer.apply_rerank_template(query, doc);  

        // 2. Tokenize
        std::vector<powerserve::Token> tokens = tokenizer.tokenize(prompt, false);
        
        float score = model.compute_rerank_score(tokens, batch_size);
        
        output.m_input_num_token += tokens.size();
        all_results.push_back({i, score});
    }

     // 4. Sort Descending
     std::sort(all_results.begin(), all_results.end(), [](const RerankResult& a, const RerankResult& b) {
        return a.score > b.score;
    });

    // 5. Top N Truncate
    if (input.m_top_n < all_results.size()) {
        all_results.resize(input.m_top_n);
    }
    
    output.m_rerank_results = std::move(all_results);
    output.m_output_num_token = 0;
    
    const size_t latency_ms = timer.elapsed_time_ms();
    POWERSERVE_LOG_INFO("Rerank finished: {} docs processed in {}ms", documents.size(), latency_ms);


    return output;
}


/*!
 * @brief Generate
 * @param[inout] context
 * @todo Streamly generation
 */
inline ModelOutput completion(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelContext &context = server_context.setup_model_for_blocking_pd(input);
    const Tokenizer &tokenizer  = *context.m_tokenizer_ptr;

    return blocking_inference(context, input, input.m_prompt);
}

inline void completion(ServerContext &server_context, ServerSession &session) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelInput &input     = session.m_input;
    const ModelContext &context = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer  = *context.m_tokenizer_ptr;

    stream_inference(context, session, input.m_prompt);
}

inline ModelOutput chat(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelContext &context    = server_context.setup_model_for_blocking_pd(input);
    const Tokenizer &tokenizer     = *context.m_tokenizer_ptr;
    const std::string input_prompt = tokenizer.apply_chat_template(input.m_history, true);

    return blocking_inference(context, input, input_prompt);
}

inline void chat(ServerContext &server_context, ServerSession &session) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelInput &input        = session.m_input;
    const ModelContext &context    = server_context.setup_model(input.m_model);
    const Tokenizer &tokenizer     = *context.m_tokenizer_ptr;
    const std::string input_prompt = tokenizer.apply_chat_template(input.m_history, true);

    stream_inference(context, session, input_prompt);
}

inline ModelOutput embedding(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    /* Parse and concat user inputs */
    const ModelContext &context = server_context.setup_model(input.m_model);
    auto model_exec_lock = lock_model_execution(context);
    const std::string input_prompt = input.m_prompt;
    return blocking_embedding(context, input, input_prompt);
}

inline ModelOutput rerank(ServerContext &server_context, const ModelInput &input) {
    using namespace powerserve;
    const ModelContext &context = server_context.setup_model(input.m_model);
    auto model_exec_lock = lock_model_execution(context);
    return blocking_rerank(context, input);
}

inline std::vector<std::string> list_models(ServerContext &server_context) {
    return server_context.list_models();
}
