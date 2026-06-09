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

#include "server_handler.hpp"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <future>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct RagRequest {
    std::string doc;
    std::string query;

    std::string mode = "sequential";
    bool enable_query_expansion = false;

    std::string generation_model;
    std::string embedding_model;
    std::string rerank_model;
    std::string expansion_model;

    size_t top_k = 20;
    size_t top_n = 5;
    size_t max_tokens = 128;
    size_t generation_decode_steps = 64;
    std::string generation_prefill_backend = "auto";
    std::string generation_decode_backend = "auto";
    float temperature = 0.2F;
};

struct RagStageMetrics {
    size_t indexing_ms = 0;
    size_t query_expand_ms = 0;
    size_t query_embedding_ms = 0;
    size_t embedding_ms = 0;
    size_t searching_ms = 0;
    size_t reranking_ms = 0;
    size_t generation_ms = 0;
    size_t total_ms = 0;
};

struct GenerationSubMetrics {
    size_t prefill_ms = 0;
    size_t decode_ms = 0;
    size_t prefill_sum_ms = 0;
    size_t decode_sum_ms = 0;
    size_t bridge_ms = 0;
    size_t kv_snapshot_ms = 0;
    size_t kv_restore_ms = 0;
    size_t kv_snapshot_bytes = 0;
};

struct DecodeTaskDebugSummary {
    std::string source;
    size_t output_tokens = 0;
    size_t output_chars = 0;
    std::string stop_reason;
    std::string text_preview;
};

struct RagResponse {
    std::string answer;
    std::string mode_requested;
    std::string mode_used;
    std::string query_used;
    std::vector<std::string> sub_queries;
    bool generation_segmented_prefill_used = false;
    size_t generation_prefill_queue_wait_ms = 0;
    size_t generation_decode_steps = 0;
    size_t decode_task_count = 0;
    std::string selected_answer_source;
    size_t candidate_count = 0;
    std::string merge_policy_version;
    std::string generation_prefill_backend_target = "auto";
    std::string generation_decode_backend_target = "auto";
    bool generation_kv_bridge_available = false;
    std::string generation_route_note;
    std::vector<DecodeTaskDebugSummary> decode_task_summaries;
    std::vector<std::string> context_chunks;
    std::vector<size_t> top_k_indices;
    std::vector<size_t> top_n_indices;
    GenerationSubMetrics generation_sub_metrics;
    RagStageMetrics metrics;
};

inline size_t next_rag_request_id() {
    static std::atomic_size_t request_counter{0};
    return request_counter.fetch_add(1, std::memory_order_relaxed);
}

struct GenerationRoutePlan {
    std::string prefill_backend_target = "auto";
    std::string decode_backend_target = "auto";
    bool kv_bridge_available = false;
    std::string route_note;
};

inline std::string normalize_backend_target(std::string backend) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    while (!backend.empty() && is_space(static_cast<unsigned char>(backend.front()))) {
        backend.erase(backend.begin());
    }
    while (!backend.empty() && is_space(static_cast<unsigned char>(backend.back()))) {
        backend.pop_back();
    }
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (backend == "cpu" || backend == "npu" || backend == "auto") {
        return backend;
    }
    return "auto";
}

inline bool is_npu_available_in_binary() {
#if defined(POWERSERVE_WITH_QNN)
    return true;
#else
    return false;
#endif
}

inline std::string normalize_model_name(std::string model_name) {
    std::transform(model_name.begin(), model_name.end(), model_name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return model_name;
}

inline bool model_has_inproc_kv_bridge(const std::string &generation_model) {
    // Current bridge capability is model-specific: Qwen3 has sync_qnn_kv_to_cpu().
    const std::string normalized = normalize_model_name(generation_model);
    return normalized.find("qwen3") != std::string::npos;
}

inline std::string resolve_prefill_backend_target(const std::string &target, bool npu_available) {
    if (target == "auto") {
        return npu_available ? "npu" : "cpu";
    }
    return target;
}

inline std::string resolve_decode_backend_target(const std::string &target, bool npu_available) {
    if (target == "auto") {
        // Keep decode on CPU by default to match current hetero intent.
        return "cpu";
    }
    if (target == "npu" && !npu_available) {
        return "cpu";
    }
    return target;
}

inline bool is_kv_bridge_available_for_route(
    const RagRequest &request,
    const std::string &resolved_prefill_backend,
    const std::string &resolved_decode_backend,
    bool npu_available
) {
    if (!npu_available) {
        return false;
    }
    if (resolved_prefill_backend != "npu" || resolved_decode_backend != "cpu") {
        return false;
    }
    return model_has_inproc_kv_bridge(request.generation_model);
}

inline GenerationRoutePlan plan_generation_route(const RagRequest &request) {
    GenerationRoutePlan plan;

    plan.prefill_backend_target = normalize_backend_target(request.generation_prefill_backend);
    plan.decode_backend_target = normalize_backend_target(request.generation_decode_backend);
    const bool npu_available = is_npu_available_in_binary();
    if (plan.prefill_backend_target == "npu" && !npu_available) {
        plan.prefill_backend_target = "auto";
        plan.route_note = "prefill_backend npu unavailable in current binary; fallback to auto";
    }
    if (plan.decode_backend_target == "npu" && !npu_available) {
        plan.decode_backend_target = "auto";
        if (!plan.route_note.empty()) {
            plan.route_note += "; ";
        }
        plan.route_note += "decode_backend npu unavailable in current binary; fallback to auto";
    }

    plan.prefill_backend_target = resolve_prefill_backend_target(plan.prefill_backend_target, npu_available);
    plan.decode_backend_target = resolve_decode_backend_target(plan.decode_backend_target, npu_available);
    plan.kv_bridge_available = is_kv_bridge_available_for_route(
        request,
        plan.prefill_backend_target,
        plan.decode_backend_target,
        npu_available
    );
    return plan;
}

inline std::string rag_trim(std::string s) {
    const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };

    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

inline std::string rag_first_line(const std::string &text) {
    const size_t pos = text.find_first_of("\r\n");
    if (pos == std::string::npos) {
        return text;
    }
    return text.substr(0, pos);
}

inline std::string rag_replace_fullwidth_semicolon(std::string s) {
    static const std::string fullwidth = "；";
    size_t pos = 0;
    while ((pos = s.find(fullwidth, pos)) != std::string::npos) {
        s.replace(pos, fullwidth.size(), ";");
        pos += 1;
    }
    return s;
}

inline std::vector<std::string> rag_split_sub_queries(
    const std::string &expanded,
    size_t max_subqueries = 3,
    size_t max_subquery_chars = 128
) {
    std::vector<std::string> out;
    std::string normalized = rag_replace_fullwidth_semicolon(rag_first_line(expanded));
    std::string current;

    auto push_token = [&](std::string token) {
        token = rag_trim(std::move(token));
        while (!token.empty() && (token.front() == '-' || token.front() == '*' || token.front() == ';')) {
            token.erase(token.begin());
            token = rag_trim(std::move(token));
        }
        if (token.empty()) {
            return;
        }
        if (token.size() > max_subquery_chars) {
            token.resize(max_subquery_chars);
            token = rag_trim(std::move(token));
        }
        if (token.empty()) {
            return;
        }
        if (std::find(out.begin(), out.end(), token) != out.end()) {
            return;
        }
        out.push_back(std::move(token));
    };

    for (const char ch : normalized) {
        if (ch == ';') {
            push_token(std::move(current));
            current.clear();
            if (out.size() >= max_subqueries) {
                break;
            }
            continue;
        }
        current.push_back(ch);
    }

    if (out.size() < max_subqueries) {
        push_token(std::move(current));
    }

    return out;
}

inline std::vector<std::string> rag_split_document(const std::string &doc) {
    std::vector<std::string> chunks;
    std::string current;
    current.reserve(doc.size());

    for (const char ch : doc) {
        current.push_back(ch);
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
            std::string trimmed = rag_trim(current);
            if (!trimmed.empty()) {
                chunks.push_back(std::move(trimmed));
            }
            current.clear();
        }
    }

    std::string tail = rag_trim(current);
    if (!tail.empty()) {
        chunks.push_back(std::move(tail));
    }

    return chunks;
}

inline std::vector<size_t> rag_search_faiss_ip(
    const std::vector<std::vector<float>> &doc_embeddings,
    const std::vector<size_t> &doc_embedding_source_indices,
    const std::vector<float> &query_embedding,
    size_t top_k
) {
    if (doc_embeddings.empty()) {
        throw std::runtime_error("document embeddings are empty");
    }
    if (query_embedding.empty()) {
        throw std::runtime_error("query embedding is empty");
    }
    if (doc_embeddings.size() != doc_embedding_source_indices.size()) {
        throw std::runtime_error("doc embedding index mapping size mismatch");
    }

    const size_t dim = query_embedding.size();
    for (const auto &emb : doc_embeddings) {
        if (emb.size() != dim) {
            throw std::runtime_error("embedding dimension mismatch");
        }
    }

    const size_t k = std::min(top_k, doc_embeddings.size());
    if (k == 0) {
        return {};
    }

    std::vector<float> database;
    database.reserve(doc_embeddings.size() * dim);
    for (const auto &emb : doc_embeddings) {
        database.insert(database.end(), emb.begin(), emb.end());
    }

    faiss::IndexFlatIP index(static_cast<faiss::Index::idx_t>(dim));
    index.add(static_cast<faiss::Index::idx_t>(doc_embeddings.size()), database.data());

    std::vector<float> distances(k);
    std::vector<faiss::Index::idx_t> labels(k);
    index.search(1, query_embedding.data(), static_cast<faiss::Index::idx_t>(k), distances.data(), labels.data());

    std::vector<size_t> top_indices;
    top_indices.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const faiss::Index::idx_t label = labels[i];
        if (label < 0) {
            continue;
        }
        const size_t dense_idx = static_cast<size_t>(label);
        if (dense_idx >= doc_embedding_source_indices.size()) {
            continue;
        }
        top_indices.push_back(doc_embedding_source_indices[dense_idx]);
    }
    return top_indices;
}

inline std::vector<size_t> rag_merge_subquery_hits(
    const std::vector<std::vector<size_t>> &per_query_hits,
    size_t top_k
) {
    std::unordered_map<size_t, float> score_by_index;
    for (const auto &hits : per_query_hits) {
        for (size_t rank = 0; rank < hits.size(); ++rank) {
            const size_t idx = hits[rank];
            score_by_index[idx] += 1.0F / (1.0F + static_cast<float>(rank));
        }
    }

    std::vector<std::pair<size_t, float>> scored_hits;
    scored_hits.reserve(score_by_index.size());
    for (const auto &[idx, score] : score_by_index) {
        scored_hits.emplace_back(idx, score);
    }

    std::sort(scored_hits.begin(), scored_hits.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });

    const size_t keep_n = std::min(top_k, scored_hits.size());
    std::vector<size_t> merged_indices;
    merged_indices.reserve(keep_n);
    for (size_t i = 0; i < keep_n; ++i) {
        merged_indices.push_back(scored_hits[i].first);
    }
    return merged_indices;
}

inline ModelInput make_generation_input(const RagRequest &request, const std::string &prompt) {
    return ModelInput{
        .m_model = request.generation_model,
        .m_prompt = prompt,
        .m_max_num_token = request.max_tokens,
        .m_temperature = request.temperature,
        .m_top_p = 1.0F,
        .m_presence_penalty = 0.0F,
        .m_frequency_penalty = 0.0F,
        .m_response_n = 1,
        .m_best_of_n = 1,
        .m_log_probs = 0,
        .stream = false,
        .m_repeat_penalty = 1.0F,
        .request_id = next_rag_request_id()
    };
}

inline void apply_generation_route_to_input(ModelInput &input, const GenerationRoutePlan &route_plan) {
    input.m_generation_route_enabled = true;
    input.m_generation_prefill_backend_target = route_plan.prefill_backend_target;
    input.m_generation_decode_backend_target = route_plan.decode_backend_target;
}

inline ModelInput make_embedding_input(const std::string &model, const std::string &text) {
    return ModelInput{
        .m_model = model,
        .m_prompt = text,
        .request_id = next_rag_request_id()
    };
}

inline ModelInput make_rerank_input(
    const RagRequest &request,
    const std::string &query,
    const std::vector<std::string> &documents
) {
    return ModelInput{
        .m_model = request.rerank_model,
        .m_prompt = query,
        .m_documents = documents,
        .m_top_n = request.top_n,
        .request_id = next_rag_request_id()
    };
}

inline std::string build_generation_prompt(const std::string &query, const std::vector<std::string> &context_chunks) {
    std::ostringstream oss;
    oss << "You are a concise and faithful QA assistant. Use only the provided context.\n";
    oss << "Context:\n";
    for (const auto &chunk : context_chunks) {
        oss << "- " << chunk << "\n";
    }
    oss << "Question: " << query << "\n";
    oss << "Answer:";
    return oss.str();
}

inline std::vector<std::string> build_generation_segments(
    const std::string &query,
    const std::vector<std::string> &sub_queries,
    const std::vector<std::string> &context_chunks
) {
    std::vector<std::string> segments;
    segments.reserve(3);

    std::ostringstream segment_1;
    segment_1 << "You are a concise and faithful QA assistant. Use only the provided context.\n";
    segment_1 << "Question: " << query << "\n";
    segments.push_back(segment_1.str());

    // Keep each sub-query as an individual segment so queue scheduling granularity
    // matches sub-query units instead of one merged hint block.
    for (const auto &sub_query : sub_queries) {
        std::ostringstream segment_2_item;
        segment_2_item << "Sub-query hint:\n";
        segment_2_item << "- " << sub_query << "\n";
        segments.push_back(segment_2_item.str());
    }

    std::ostringstream segment_3;
    segment_3 << "Context:\n";
    for (const auto &chunk : context_chunks) {
        segment_3 << "- " << chunk << "\n";
    }
    segment_3 << "Answer:";
    segments.push_back(segment_3.str());

    return segments;
}

inline std::vector<GenerationDecodeTask> build_generation_decode_tasks(
    const std::string &query,
    const std::vector<std::string> &sub_queries,
    const std::vector<std::string> &context_chunks,
    size_t max_decode_steps
) {
    std::vector<GenerationDecodeTask> tasks;
    tasks.reserve(1 + sub_queries.size());

    GenerationDecodeTask original_task;
    original_task.query_type = "original";
    original_task.max_decode_steps = max_decode_steps;
    original_task.input_segments = build_generation_segments(query, sub_queries, context_chunks);
    tasks.push_back(std::move(original_task));

    for (const auto &sub_query : sub_queries) {
        GenerationDecodeTask task;
        task.query_type = "subquery";
        task.max_decode_steps = max_decode_steps;
        task.input_segments = build_generation_segments(sub_query, {}, context_chunks);
        tasks.push_back(std::move(task));
    }

    return tasks;
}

struct GenerationMergeResult {
    size_t selected_candidate_idx = 0;
    std::string selected_source = "original";
    std::string answer;
    std::string merge_policy_version = "v1-rule";
};

inline GenerationMergeResult merge_generation_candidates_v1(const std::vector<GenerationDecodeCandidate> &candidates) {
    GenerationMergeResult result;
    if (candidates.empty()) {
        return result;
    }

    auto is_candidate_valid = [](const GenerationDecodeCandidate &candidate) {
        return !rag_trim(candidate.output.m_text).empty();
    };

    const size_t min_original_chars = 24;

    size_t original_idx = 0;
    bool found_original = false;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].source == "original") {
            original_idx = i;
            found_original = true;
            break;
        }
    }

    size_t best_subquery_idx = original_idx;
    bool found_valid_subquery = false;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].source != "subquery" || !is_candidate_valid(candidates[i])) {
            continue;
        }
        if (!found_valid_subquery || candidates[i].output.m_text.size() > candidates[best_subquery_idx].output.m_text.size()) {
            best_subquery_idx = i;
            found_valid_subquery = true;
        }
    }

    if (!found_original) {
        result.selected_candidate_idx = found_valid_subquery ? best_subquery_idx : 0;
    } else {
        const bool original_valid = is_candidate_valid(candidates[original_idx]);
        const bool original_too_short = candidates[original_idx].output.m_text.size() < min_original_chars;
        const bool original_stop_abnormal =
            !candidates[original_idx].output.m_stop_reason.has_value() ||
            (candidates[original_idx].output.m_stop_reason.value() != "stop" &&
             candidates[original_idx].output.m_stop_reason.value() != "length");

        if ((!original_valid || original_too_short || original_stop_abnormal) && found_valid_subquery) {
            result.selected_candidate_idx = best_subquery_idx;
        } else {
            result.selected_candidate_idx = original_idx;
        }
    }

    result.selected_source = candidates[result.selected_candidate_idx].source;
    result.answer = candidates[result.selected_candidate_idx].output.m_text;
    return result;
}

inline std::string maybe_expand_rag_query(
    ServerContext &server_context,
    const RagRequest &request,
    const GenerationRoutePlan &generation_route_plan,
    size_t &query_expand_ms
) {
    using namespace powerserve;

    query_expand_ms = 0;
    if (!request.enable_query_expansion) {
        return request.query;
    }

    Timer stage_timer;
    const std::string model_for_expand = request.expansion_model.empty() ? request.generation_model : request.expansion_model;
    const std::string expand_prompt =
        "You are a query rewriter for a retrieval system. "
    "Given the user's query, generate three different sub-queries. "
    "Each sub-query should focus on a distinct aspect or phrasing of the original query. "
    "Return the three sub-queries on a single line, separated by semicolon(;). "
    "Do NOT use numbers. "
    "Do NOT answer the query. "
    "Query: " + request.query + "\n"
    "Your output:";

    ModelInput expand_input = make_generation_input(request, expand_prompt);
    apply_generation_route_to_input(expand_input, generation_route_plan);
    expand_input.m_model = model_for_expand;
    expand_input.m_temperature = 0.2F;
    expand_input.m_max_num_token = std::min<size_t>(request.max_tokens, size_t{96});

    const ModelOutput expand_output = completion(server_context, expand_input);
    printf("Query expansion output: %s\n", expand_output.m_text.c_str());
    query_expand_ms = stage_timer.elapsed_time_ms();

    const std::string candidate = rag_trim(rag_first_line(expand_output.m_text));
    return candidate.empty() ? request.query : candidate;
}

inline void log_rag_stage_metrics(const RagResponse &response) {
    POWERSERVE_LOG_INFO(
        "rag stage metrics (mode={}): indexing(split+doc_embedding)={}ms, query_expand={}ms, query_embedding={}ms, embedding={}ms, searching={}ms, reranking={}ms, generation={}ms, total={}ms, generation_sub(prefill={}ms, decode={}ms, prefill_sum={}ms, decode_sum={}ms, bridge={}ms, kv_snapshot={}ms, kv_restore={}ms, kv_snapshot_bytes={})",
        response.mode_used,
        response.metrics.indexing_ms,
        response.metrics.query_expand_ms,
        response.metrics.query_embedding_ms,
        response.metrics.embedding_ms,
        response.metrics.searching_ms,
        response.metrics.reranking_ms,
        response.metrics.generation_ms,
        response.metrics.total_ms,
        response.generation_sub_metrics.prefill_ms,
        response.generation_sub_metrics.decode_ms,
        response.generation_sub_metrics.prefill_sum_ms,
        response.generation_sub_metrics.decode_sum_ms,
        response.generation_sub_metrics.bridge_ms,
        response.generation_sub_metrics.kv_snapshot_ms,
        response.generation_sub_metrics.kv_restore_ms,
        response.generation_sub_metrics.kv_snapshot_bytes
    );
}

inline RagResponse run_rag_sequential(ServerContext &server_context, const RagRequest &request) {
    using namespace powerserve;

    if (request.doc.empty()) {
        throw std::invalid_argument("'doc' must not be empty");
    }
    if (request.query.empty()) {
        throw std::invalid_argument("'query' must not be empty");
    }
    if (request.generation_model.empty() || request.embedding_model.empty() || request.rerank_model.empty()) {
        throw std::invalid_argument("'generation_model', 'embedding_model', and 'rerank_model' are required");
    }

    RagResponse response;
    response.mode_requested = request.mode;
    response.mode_used = request.mode == "hetero_parallel" ? "sequential" : request.mode;
    response.query_used = request.query;
    const GenerationRoutePlan generation_route_plan = plan_generation_route(request);
    response.generation_prefill_backend_target = generation_route_plan.prefill_backend_target;
    response.generation_decode_backend_target = generation_route_plan.decode_backend_target;
    response.generation_kv_bridge_available = generation_route_plan.kv_bridge_available;
    response.generation_route_note = generation_route_plan.route_note;

    Timer total_timer;

    // 1) Indexing (doc split + doc embedding)
    Timer stage_timer;
    const std::vector<std::string> chunks = rag_split_document(request.doc);
    const size_t split_ms = stage_timer.elapsed_time_ms();
    if (chunks.empty()) {
        throw std::invalid_argument("'doc' does not contain valid chunks after indexing");
    }

    stage_timer = Timer{};
    size_t doc_embedding_ms = 0;
    std::vector<std::vector<float>> doc_embeddings;
    std::vector<size_t> doc_embedding_source_indices;
    doc_embeddings.reserve(chunks.size());
    doc_embedding_source_indices.reserve(chunks.size());
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        const auto &chunk = chunks[chunk_idx];
        const ModelOutput doc_embedding_out = embedding(server_context, make_embedding_input(request.embedding_model, chunk));
        if (doc_embedding_out.m_embedding.empty()) {
            continue;
        }
        doc_embeddings.push_back(doc_embedding_out.m_embedding);
        doc_embedding_source_indices.push_back(chunk_idx);
    }
    doc_embedding_ms = stage_timer.elapsed_time_ms();
    response.metrics.indexing_ms = split_ms + doc_embedding_ms;
    response.metrics.embedding_ms = doc_embedding_ms;
    if (doc_embeddings.empty()) {
        throw std::runtime_error("all document embeddings are empty");
    }

    // 2) Query expansion (optional)
    const std::string expanded_query = maybe_expand_rag_query(
        server_context,
        request,
        generation_route_plan,
        response.metrics.query_expand_ms
    );

    // Phase A: in sequential mode, cache split sub-queries for later phases.
    if (request.enable_query_expansion) {
        // Keep split step for pipeline shape, then force fixed sub-queries for temporary experiments.
        const auto parsed_sub_queries = rag_split_sub_queries(expanded_query);
        (void)parsed_sub_queries;
        response.sub_queries = {
            "OpenAI 在技术方面的发展中体现了哪些权衡取舍?",
            "OpenAI 在商业方面的发展中体现了哪些权衡取舍?",
            "OpenAI 在安全方面的发展中体现了哪些权衡取舍?"
        };
    }
    const std::vector<std::string> retrieval_queries =
        response.sub_queries.empty() ? std::vector<std::string>{expanded_query} : response.sub_queries;
    response.query_used = request.query;

    // 3) Query embeddings (all sub-queries first)
    stage_timer = Timer{};
    std::vector<std::vector<float>> query_embeddings;
    query_embeddings.reserve(retrieval_queries.size());
    for (const auto &sub_query : retrieval_queries) {
        const ModelOutput query_embedding_out = embedding(server_context, make_embedding_input(request.embedding_model, sub_query));
        if (query_embedding_out.m_embedding.empty()) {
            continue;
        }
        query_embeddings.push_back(query_embedding_out.m_embedding);
    }
    if (query_embeddings.empty()) {
        throw std::runtime_error("all query embeddings are empty");
    }
    response.metrics.query_embedding_ms = stage_timer.elapsed_time_ms();
    response.metrics.embedding_ms += response.metrics.query_embedding_ms;

    // 4) Searching (all sub-queries, then merge/dedup)
    stage_timer = Timer{};
    std::vector<std::vector<size_t>> per_query_top_indices;
    per_query_top_indices.reserve(query_embeddings.size());
    for (const auto &query_embedding : query_embeddings) {
        per_query_top_indices.push_back(rag_search_faiss_ip(
            doc_embeddings,
            doc_embedding_source_indices,
            query_embedding,
            request.top_k
        ));
    }
    const std::vector<size_t> merged_top_indices = rag_merge_subquery_hits(per_query_top_indices, request.top_k);

    const size_t actual_top_k = merged_top_indices.size();
    std::vector<std::string> top_k_docs;
    top_k_docs.reserve(actual_top_k);
    for (size_t i = 0; i < actual_top_k; ++i) {
        response.top_k_indices.push_back(merged_top_indices[i]);
        top_k_docs.push_back(chunks[merged_top_indices[i]]);
    }
    response.metrics.searching_ms = stage_timer.elapsed_time_ms();
    if (top_k_docs.empty()) {
        throw std::runtime_error("retrieval returns empty top_k docs");
    }

    // 5) Reranking
    stage_timer = Timer{};
    const ModelOutput rerank_out = rerank(server_context, make_rerank_input(request, request.query, top_k_docs));

    std::vector<std::string> selected_context;
    for (const auto &item : rerank_out.m_rerank_results) {
        if (item.index >= top_k_docs.size()) {
            continue;
        }
        response.top_n_indices.push_back(response.top_k_indices[item.index]);
        selected_context.push_back(top_k_docs[item.index]);
    }

    if (selected_context.empty()) {
        const size_t fallback_n = std::min(request.top_n, top_k_docs.size());
        for (size_t i = 0; i < fallback_n; ++i) {
            response.top_n_indices.push_back(response.top_k_indices[i]);
            selected_context.push_back(top_k_docs[i]);
        }
    }
    response.metrics.reranking_ms = stage_timer.elapsed_time_ms();

    // 6) Generation
    stage_timer = Timer{};
    const std::vector<GenerationDecodeTask> generation_tasks = build_generation_decode_tasks(
        request.query,
        response.sub_queries,
        selected_context,
        request.generation_decode_steps
    );

    std::vector<GenerationDecodeCandidate> generation_candidates;
    generation_candidates.reserve(generation_tasks.size());

    bool segmented_prefill_used = false;
    try {
        for (const auto &task : generation_tasks) {
            const std::string task_prompt = task.input_segments.empty()
                ? build_generation_prompt(request.query, selected_context)
                : task.input_segments.front();

            ModelInput generation_input = make_generation_input(request, task_prompt);
            apply_generation_route_to_input(generation_input, generation_route_plan);
            const ModelContext &generation_context = server_context.setup_model_for_blocking_pd(generation_input);
            GenerationDecodeCandidate candidate = blocking_inference_segmented_prefill_decode_task(
                server_context,
                generation_context,
                generation_input,
                task
            );
            generation_candidates.push_back(std::move(candidate));
        }

        const GenerationMergeResult merge_result = merge_generation_candidates_v1(generation_candidates);
        const GenerationDecodeCandidate &selected_candidate = generation_candidates.at(merge_result.selected_candidate_idx);
        response.answer = selected_candidate.output.m_text;
        response.generation_prefill_queue_wait_ms = 0;
        for (const auto &candidate : generation_candidates) {
            response.generation_prefill_queue_wait_ms += candidate.queue_wait_ms;
        }
        response.decode_task_count = generation_tasks.size();
        response.candidate_count = generation_candidates.size();
        response.selected_answer_source = merge_result.selected_source;
        response.merge_policy_version = merge_result.merge_policy_version;
        response.decode_task_summaries.clear();
        response.decode_task_summaries.reserve(generation_candidates.size());
        for (const auto &candidate : generation_candidates) {
            std::string preview = candidate.output.m_text;
            constexpr size_t kMaxPreviewChars = 200;
            if (preview.size() > kMaxPreviewChars) {
                preview.resize(kMaxPreviewChars);
                remove_incomplete_utf8_char(preview);
            }
            response.decode_task_summaries.push_back({
                .source = candidate.source,
                .output_tokens = candidate.output.m_output_num_token > 1 ? candidate.output.m_output_num_token - 1 : 0,
                .output_chars = candidate.output.m_text.size(),
                .stop_reason = candidate.output.m_stop_reason.value_or("unknown"),
                .text_preview = std::move(preview),
            });
        }

        segmented_prefill_used = true;
    } catch (...) {
        const std::string generation_prompt = build_generation_prompt(request.query, selected_context);
        ModelInput fallback_generation_input = make_generation_input(request, generation_prompt);
        apply_generation_route_to_input(fallback_generation_input, generation_route_plan);
        const ModelOutput generation_out = completion(server_context, fallback_generation_input);
        response.answer = generation_out.m_text;
        response.decode_task_count = 1;
        response.selected_answer_source = "original";
        response.candidate_count = 1;
        response.merge_policy_version = "v1-rule";
        response.decode_task_summaries = {
            DecodeTaskDebugSummary{
                .source = "original",
                .output_tokens = generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0,
                .output_chars = generation_out.m_text.size(),
                .stop_reason = generation_out.m_stop_reason.value_or("unknown"),
                .text_preview = generation_out.m_text,
            }
        };

        response.generation_decode_steps =
            generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0;
    }

    response.generation_segmented_prefill_used = segmented_prefill_used;
    if (segmented_prefill_used && !generation_candidates.empty()) {
        size_t selected_candidate_idx = 0;
        for (size_t i = 0; i < generation_candidates.size(); ++i) {
            if (generation_candidates[i].source == response.selected_answer_source) {
                selected_candidate_idx = i;
                break;
            }
        }
        response.generation_decode_steps =
            generation_candidates[selected_candidate_idx].output.m_output_num_token > 1
            ? generation_candidates[selected_candidate_idx].output.m_output_num_token - 1
            : 0;
    }
    response.context_chunks = std::move(selected_context);
    response.metrics.generation_ms = stage_timer.elapsed_time_ms();

    response.metrics.total_ms = total_timer.elapsed_time_ms();
    log_rag_stage_metrics(response);
    return response;
}

inline RagResponse run_rag_hetero_parallel(ServerContext &server_context, const RagRequest &request) {
    using namespace powerserve;

    if (request.doc.empty()) {
        throw std::invalid_argument("'doc' must not be empty");
    }
    if (request.query.empty()) {
        throw std::invalid_argument("'query' must not be empty");
    }
    if (request.generation_model.empty() || request.embedding_model.empty() || request.rerank_model.empty()) {
        throw std::invalid_argument("'generation_model', 'embedding_model', and 'rerank_model' are required");
    }

    struct DocBranchOutput {
        std::vector<std::string> chunks;
        std::vector<std::vector<float>> doc_embeddings;
        std::vector<size_t> doc_embedding_source_indices;
        size_t indexing_ms = 0;
        size_t doc_embedding_ms = 0;
    };

    struct QueryBranchOutput {
        std::string expanded_query;
        std::vector<std::string> sub_queries;
        size_t query_expand_ms = 0;
    };

    RagResponse response;
    response.mode_requested = request.mode;
    response.mode_used = "hetero_parallel";
    response.query_used = request.query;
    const GenerationRoutePlan generation_route_plan = plan_generation_route(request);
    response.generation_prefill_backend_target = generation_route_plan.prefill_backend_target;
    response.generation_decode_backend_target = generation_route_plan.decode_backend_target;
    response.generation_kv_bridge_available = generation_route_plan.kv_bridge_available;
    response.generation_route_note = generation_route_plan.route_note;

    Timer total_timer;
    POWERSERVE_ASSERT(server_context.scheduler2 != nullptr);
    POWERSERVE_ASSERT(server_context.kv_cache_manager != nullptr);

    DocBranchOutput doc_branch;
    QueryBranchOutput query_branch;
    const size_t retrieval_branch_count = request.enable_query_expansion ? 3 : 1;
    const size_t generation_task_count = request.enable_query_expansion ? 4 : 1;

    std::vector<std::vector<float>> query_embeddings(retrieval_branch_count);
    std::vector<std::vector<size_t>> per_query_top_indices(retrieval_branch_count);
    std::vector<size_t> merged_top_indices;
    std::vector<std::string> top_k_docs;
    std::vector<size_t> top_k_indices_collected;
    std::vector<size_t> top_n_indices_collected;
    std::vector<std::string> selected_context;

    std::atomic_size_t query_embedding_ms_acc{0};
    std::atomic_size_t searching_ms_acc{0};
    std::atomic_size_t reranking_ms_acc{0};
    std::atomic_size_t generation_prefill_ms_acc{0};
    std::atomic_size_t generation_decode_ms_acc{0};
    std::atomic_size_t generation_bridge_ms_acc{0};
    std::atomic_size_t generation_snapshot_ms_acc{0};
    std::atomic_size_t generation_restore_ms_acc{0};
    std::atomic_size_t generation_snapshot_bytes_acc{0};
    std::atomic_long generation_prefill_begin_ms{-1};
    std::atomic_long generation_prefill_end_ms{-1};
    std::atomic_long generation_decode_begin_ms{-1};
    std::atomic_long generation_decode_end_ms{-1};

    std::vector<GenerationDecodeTask> generation_tasks(generation_task_count);
    std::vector<GenerationDecodeCandidate> generation_candidates(generation_task_count);
    std::vector<ModelInput> generation_inputs(generation_task_count);
    std::vector<bool> generation_input_ready(generation_task_count, false);
    std::vector<const ModelContext *> generation_contexts(generation_task_count, nullptr);
    std::vector<BlockingPrefillResult> prefill_results(generation_task_count);
    std::vector<std::unique_ptr<powerserve::SamplerChain>> prefill_samplers(generation_task_count);
    std::vector<std::unique_ptr<powerserve::ggml::GGMLKV::Snapshot>> prefill_kv_snapshots(generation_task_count);

    bool segmented_prefill_used = false;
    std::atomic_long generation_start_ms{-1};

    const bool npu_available =
#if defined(POWERSERVE_WITH_QNN)
        true;
#else
        false;
#endif
    const auto prefill_route = server_context.backend_router.route_for_generation_prefill(
        generation_route_plan.prefill_backend_target,
        npu_available
    );
    const auto decode_route = server_context.backend_router.route_for_generation_decode(
        generation_route_plan.decode_backend_target,
        npu_available
    );

    constexpr size_t indexing_node_id = 1;
    constexpr size_t query_expand_node_id = 2;
    constexpr size_t reranking_node_id = 9000;
    constexpr size_t generation_prefill_base_node_id = 20000;
    constexpr size_t generation_merge_node_id = 26000;

    std::vector<powerserve::Scheduler2DagNode> dag_nodes;
    dag_nodes.reserve(2 + retrieval_branch_count * 2 + 1 + generation_task_count * 2 + 1);

    dag_nodes.push_back({
        .node_id = indexing_node_id,
        .type = powerserve::Scheduler2TaskType::UNKNOWN,
        .request_id = next_rag_request_id(),
        .backend = powerserve::BackendKind::CPU,
        .dependencies = {},
        .fn = [&]() {
            DocBranchOutput out;

            Timer stage_timer;
            out.chunks = rag_split_document(request.doc);
            const size_t split_ms = stage_timer.elapsed_time_ms();
            if (out.chunks.empty()) {
                throw std::invalid_argument("'doc' does not contain valid chunks after indexing");
            }

            stage_timer = Timer{};
            out.doc_embeddings.reserve(out.chunks.size());
            out.doc_embedding_source_indices.reserve(out.chunks.size());
            for (size_t chunk_idx = 0; chunk_idx < out.chunks.size(); ++chunk_idx) {
                const auto &chunk = out.chunks[chunk_idx];
                const ModelOutput doc_embedding_out =
                    embedding(server_context, make_embedding_input(request.embedding_model, chunk));
                if (doc_embedding_out.m_embedding.empty()) {
                    continue;
                }
                out.doc_embeddings.push_back(doc_embedding_out.m_embedding);
                out.doc_embedding_source_indices.push_back(chunk_idx);
            }
            out.doc_embedding_ms = stage_timer.elapsed_time_ms();
            out.indexing_ms = split_ms + out.doc_embedding_ms;

            if (out.doc_embeddings.empty()) {
                throw std::runtime_error("all document embeddings are empty");
            }
            doc_branch = std::move(out);
        },
        .debug_name = "indexing",
    });

    dag_nodes.push_back({
        .node_id = query_expand_node_id,
        .type = powerserve::Scheduler2TaskType::UNKNOWN,
        .request_id = next_rag_request_id(),
        .backend = powerserve::BackendKind::CPU,
        .dependencies = {},
        .fn = [&]() {
            QueryBranchOutput out;
            const GenerationRoutePlan route_plan = plan_generation_route(request);
            out.expanded_query = maybe_expand_rag_query(server_context, request, route_plan, out.query_expand_ms);
            if (request.enable_query_expansion) {
                const auto parsed_sub_queries = rag_split_sub_queries(out.expanded_query);
                (void)parsed_sub_queries;
                out.sub_queries = {
                    "OpenAI 在技术方面的发展中体现了哪些权衡取舍?",
                    "OpenAI 在商业方面的发展中体现了哪些权衡取舍?",
                    "OpenAI 在安全方面的发展中体现了哪些权衡取舍?"
                };
            }
            query_branch = std::move(out);
        },
        .debug_name = "query_expand",
    });

    std::vector<size_t> search_node_ids;
    search_node_ids.reserve(retrieval_branch_count);
    for (size_t i = 0; i < retrieval_branch_count; ++i) {
        const size_t embedding_node_id = 1000 + i * 2;
        const size_t searching_node_id = embedding_node_id + 1;
        search_node_ids.push_back(searching_node_id);

        dag_nodes.push_back({
            .node_id = embedding_node_id,
            .type = powerserve::Scheduler2TaskType::UNKNOWN,
            .request_id = next_rag_request_id(),
            .backend = powerserve::BackendKind::CPU,
            .dependencies = {query_expand_node_id},
            .fn = [&, i]() {
                Timer timer;
                const std::string retrieval_query = request.enable_query_expansion
                    ? query_branch.sub_queries.at(i)
                    : query_branch.expanded_query;
                const ModelOutput query_embedding_out =
                    embedding(server_context, make_embedding_input(request.embedding_model, retrieval_query));
                if (query_embedding_out.m_embedding.empty()) {
                    throw std::runtime_error("query embedding is empty");
                }
                query_embeddings[i] = query_embedding_out.m_embedding;
                query_embedding_ms_acc.fetch_add(timer.elapsed_time_ms(), std::memory_order_relaxed);
            },
            .debug_name = "query_embedding_" + std::to_string(i + 1),
        });

        dag_nodes.push_back({
            .node_id = searching_node_id,
            .type = powerserve::Scheduler2TaskType::UNKNOWN,
            .request_id = next_rag_request_id(),
            .backend = powerserve::BackendKind::CPU,
            .dependencies = {embedding_node_id, indexing_node_id},
            .fn = [&, i]() {
                Timer timer;
                per_query_top_indices[i] = rag_search_faiss_ip(
                    doc_branch.doc_embeddings,
                    doc_branch.doc_embedding_source_indices,
                    query_embeddings[i],
                    request.top_k
                );
                searching_ms_acc.fetch_add(timer.elapsed_time_ms(), std::memory_order_relaxed);
            },
            .debug_name = "searching_" + std::to_string(i + 1),
        });
    }

    std::vector<size_t> reranking_deps;
    reranking_deps.reserve(1 + search_node_ids.size());
    reranking_deps.push_back(indexing_node_id);
    reranking_deps.insert(reranking_deps.end(), search_node_ids.begin(), search_node_ids.end());
    dag_nodes.push_back({
        .node_id = reranking_node_id,
        .type = powerserve::Scheduler2TaskType::UNKNOWN,
        .request_id = next_rag_request_id(),
        .backend = powerserve::BackendKind::CPU,
        .dependencies = std::move(reranking_deps),
        .fn = [&]() {
            Timer merge_timer;
            merged_top_indices = rag_merge_subquery_hits(per_query_top_indices, request.top_k);
            const size_t actual_top_k = merged_top_indices.size();
            top_k_docs.clear();
            top_k_indices_collected.clear();
            top_k_docs.reserve(actual_top_k);
            top_k_indices_collected.reserve(actual_top_k);
            for (size_t i = 0; i < actual_top_k; ++i) {
                top_k_indices_collected.push_back(merged_top_indices[i]);
                top_k_docs.push_back(doc_branch.chunks[merged_top_indices[i]]);
            }
            searching_ms_acc.fetch_add(merge_timer.elapsed_time_ms(), std::memory_order_relaxed);

            if (top_k_docs.empty()) {
                throw std::runtime_error("retrieval returns empty top_k docs");
            }

            Timer rerank_timer;
            const ModelOutput rerank_out = rerank(server_context, make_rerank_input(request, request.query, top_k_docs));

            selected_context.clear();
            top_n_indices_collected.clear();
            selected_context.reserve(rerank_out.m_rerank_results.size());
            top_n_indices_collected.reserve(rerank_out.m_rerank_results.size());
            for (const auto &item : rerank_out.m_rerank_results) {
                if (item.index >= top_k_docs.size()) {
                    continue;
                }
                top_n_indices_collected.push_back(top_k_indices_collected[item.index]);
                selected_context.push_back(top_k_docs[item.index]);
            }
            if (selected_context.empty()) {
                const size_t fallback_n = std::min(request.top_n, top_k_docs.size());
                for (size_t i = 0; i < fallback_n; ++i) {
                    top_n_indices_collected.push_back(top_k_indices_collected[i]);
                    selected_context.push_back(top_k_docs[i]);
                }
            }
            reranking_ms_acc.store(rerank_timer.elapsed_time_ms(), std::memory_order_relaxed);

            generation_tasks = build_generation_decode_tasks(
                request.query,
                query_branch.sub_queries,
                selected_context,
                request.generation_decode_steps
            );
            if (generation_tasks.size() != generation_task_count) {
                throw std::runtime_error("generation task count mismatch while building big dag");
            }
            for (size_t i = 0; i < generation_task_count; ++i) {
                const std::string task_prompt = generation_tasks[i].input_segments.empty()
                    ? build_generation_prompt(request.query, selected_context)
                    : generation_tasks[i].input_segments.front();
                ModelInput generation_input = make_generation_input(request, task_prompt);
                apply_generation_route_to_input(generation_input, generation_route_plan);
                generation_contexts[i] = &server_context.setup_model_for_blocking_pd(generation_input);
                generation_inputs[i] = std::move(generation_input);
                generation_input_ready[i] = true;
            }
        },
        .debug_name = "reranking",
    });

    std::vector<size_t> generation_decode_node_ids;
    generation_decode_node_ids.reserve(generation_task_count);
    for (size_t i = 0; i < generation_task_count; ++i) {
        const size_t prefill_node_id = generation_prefill_base_node_id + i * 2;
        const size_t decode_node_id = prefill_node_id + 1;
        generation_decode_node_ids.push_back(decode_node_id);

        dag_nodes.push_back({
            .node_id = prefill_node_id,
            .type = powerserve::Scheduler2TaskType::GENERATION_PREFILL,
            .request_id = next_rag_request_id(),
            .backend = prefill_route.backend,
            .dependencies = {reranking_node_id},
            .fn = [&, i]() {
                Timer prefill_timer;
                const long now_ms = static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count()
                );
                long expected = -1;
                (void)generation_start_ms.compare_exchange_strong(expected, now_ms);
                expected = -1;
                (void)generation_prefill_begin_ms.compare_exchange_strong(expected, now_ms);

                POWERSERVE_ASSERT(generation_contexts[i] != nullptr);
                const ModelContext &context = *generation_contexts[i];
                const ModelInput &input = generation_inputs[i];
                const GenerationDecodeTask &decode_task = generation_tasks[i];
                const auto &tokenizer = *context.m_tokenizer_ptr;
                auto model_exec_lock = lock_model_execution(context);

                auto sampler_config = context.m_config.hyper_params.sampler_config;
                sampler_config.temperature = input.m_temperature;
                sampler_config.penalty_freq = input.m_frequency_penalty;
                sampler_config.penalty_present = input.m_presence_penalty;
                sampler_config.penalty_repeat = input.m_repeat_penalty;
                sampler_config.top_p = input.m_top_p;
                prefill_samplers[i] = std::make_unique<powerserve::SamplerChain>(sampler_config, tokenizer);
                POWERSERVE_ASSERT(prefill_samplers[i] != nullptr);

                POWERSERVE_LOG_DEBUG(
                    "scheduler2 dag prefill start: request_id={}, backend={}",
                    input.request_id,
                    BackendRouter::backend_name(prefill_route.backend)
                );
                if (input.m_generation_route_enabled) {
                    if (!set_generation_backend_route(context, BackendRouter::backend_name(prefill_route.backend))) {
                        POWERSERVE_LOG_WARN(
                            "scheduler2 dag prefill backend route fallback to cpu, request_id={}, target={}",
                            input.request_id,
                            input.m_generation_prefill_backend_target
                        );
                        (void)set_generation_backend_route(context, "cpu");
                    }
                }

                std::string model_id = context.m_model_ptr->m_config->model_id;
                const size_t kv_position_begin = context.m_model_ptr->m_platform->get_kv_position(model_id);
                BlockingPrefillResult prefill_result = run_blocking_prefill_segmented(
                    context,
                    input,
                    decode_task.input_segments,
                    *prefill_samplers[i]
                );
                PrefillArtifact artifact = build_prefill_artifact(context, input, prefill_result, kv_position_begin);
                server_context.kv_cache_manager->put({
                    .request_id = artifact.request_id,
                    .model_id = artifact.model_id,
                    .producer_backend = BackendRouter::backend_name(prefill_route.backend),
                    .kv_begin = artifact.kv_position_begin,
                    .kv_end = artifact.kv_position_end,
                    .prefill_tokens_total = artifact.prefill_tokens_total,
                });

                if (decode_route.backend == BackendKind::CPU) {
                    auto ggml_iter = context.m_model_ptr->m_platform->ggml_backends.find(artifact.model_id);
                    POWERSERVE_ASSERT(ggml_iter != context.m_model_ptr->m_platform->ggml_backends.end());
                    POWERSERVE_ASSERT(ggml_iter->second && ggml_iter->second->m_kv);

                    Timer snapshot_timer;
                    // Keep prefix-safe snapshot semantics while reducing copy size to valid tokens.
                    prefill_kv_snapshots[i] = ggml_iter->second->m_kv->save_snapshot(0, artifact.kv_position_end);
                    const size_t snapshot_ms = snapshot_timer.elapsed_time_ms();
                    size_t snapshot_bytes = 0;
                    if (prefill_kv_snapshots[i] != nullptr) {
                        for (const auto &buf : prefill_kv_snapshots[i]->key_buffer) {
                            snapshot_bytes += buf.size() * sizeof(float);
                        }
                        for (const auto &buf : prefill_kv_snapshots[i]->value_buffer) {
                            snapshot_bytes += buf.size() * sizeof(float);
                        }
                    }
                    generation_snapshot_ms_acc.fetch_add(snapshot_ms, std::memory_order_relaxed);
                    generation_snapshot_bytes_acc.fetch_add(snapshot_bytes, std::memory_order_relaxed);
                    POWERSERVE_LOG_INFO(
                        "scheduler2 dag kv snapshot: request_id={}, copied_tokens={}, bytes={}, cost_ms={}",
                        input.request_id,
                        artifact.kv_position_end,
                        snapshot_bytes,
                        snapshot_ms
                    );
                }

                prefill_results[i] = std::move(prefill_result);
                generation_prefill_ms_acc.fetch_add(prefill_timer.elapsed_time_ms(), std::memory_order_relaxed);
                const long prefill_end_ms = static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count()
                );
                long previous_prefill_end = generation_prefill_end_ms.load(std::memory_order_relaxed);
                while (previous_prefill_end < prefill_end_ms &&
                       !generation_prefill_end_ms.compare_exchange_weak(
                           previous_prefill_end,
                           prefill_end_ms,
                           std::memory_order_relaxed,
                           std::memory_order_relaxed
                       )) {
                }
                POWERSERVE_LOG_DEBUG(
                    "scheduler2 dag prefill end: request_id={}, backend={}",
                    input.request_id,
                    BackendRouter::backend_name(prefill_route.backend)
                );
            },
            .debug_name = "generation_prefill_" + std::to_string(i),
        });

        dag_nodes.push_back({
            .node_id = decode_node_id,
            .type = powerserve::Scheduler2TaskType::GENERATION_DECODE,
            .request_id = next_rag_request_id(),
            .backend = decode_route.backend,
            .dependencies = {prefill_node_id},
            .fn = [&, i]() {
                Timer decode_timer;
                const long decode_begin_ms = static_cast<long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count()
                );
                long expected = -1;
                (void)generation_decode_begin_ms.compare_exchange_strong(expected, decode_begin_ms);
                POWERSERVE_ASSERT(generation_contexts[i] != nullptr);
                const ModelContext &context = *generation_contexts[i];
                const ModelInput &input = generation_inputs[i];
                const GenerationDecodeTask &decode_task = generation_tasks[i];
                const auto cache_record = server_context.kv_cache_manager->get(input.request_id);
                if (!cache_record.has_value()) {
                    throw std::runtime_error("scheduler2 dag missing kv cache record for decode");
                }
                if (cache_record->producer_backend == "npu" && decode_route.backend == BackendKind::CPU) {
                    Timer bridge_timer;
                    const bool bridged = server_context.kv_cache_manager->bridge_to_cpu(input.request_id);
                    const size_t bridge_ms = bridge_timer.elapsed_time_ms();
                    generation_bridge_ms_acc.fetch_add(bridge_ms, std::memory_order_relaxed);
                    if (bridged) {
                        POWERSERVE_LOG_INFO(
                            "scheduler2 dag kv bridge hook: request_id={}, prefill_backend=npu, decode_backend=cpu",
                            input.request_id
                        );
                    } else {
                        POWERSERVE_LOG_WARN("scheduler2 dag kv bridge hook failed: request_id={}", input.request_id);
                    }
                }

                try {
                    const auto &tokenizer = *context.m_tokenizer_ptr;
                    auto model_exec_lock = lock_model_execution(context);
                    POWERSERVE_LOG_DEBUG(
                        "scheduler2 dag decode start: request_id={}, backend={}",
                        input.request_id,
                        BackendRouter::backend_name(decode_route.backend)
                    );

                    if (input.m_generation_route_enabled) {
                        if (!set_generation_backend_route(context, BackendRouter::backend_name(decode_route.backend))) {
                            POWERSERVE_LOG_WARN(
                                "scheduler2 dag decode backend route fallback to cpu, request_id={}, target={}",
                                input.request_id,
                                input.m_generation_decode_backend_target
                            );
                            (void)set_generation_backend_route(context, "cpu");
                        }
                    }

                    std::unique_ptr<powerserve::ggml::GGMLKV> private_ggml_kv;
                    if (decode_route.backend == BackendKind::CPU) {
                        if (prefill_kv_snapshots[i] == nullptr) {
                            throw std::runtime_error("scheduler2 dag decode missing kv snapshot");
                        }
                        auto ggml_iter = context.m_model_ptr->m_platform->ggml_backends.find(cache_record->model_id);
                        POWERSERVE_ASSERT(ggml_iter != context.m_model_ptr->m_platform->ggml_backends.end());
                        POWERSERVE_ASSERT(ggml_iter->second && ggml_iter->second->m_kv);

                        private_ggml_kv = std::make_unique<powerserve::ggml::GGMLKV>(ggml_iter->second->m_kv->m_config);
                        Timer restore_timer;
                        private_ggml_kv->restore_snapshot(*prefill_kv_snapshots[i]);
                        const size_t restore_ms = restore_timer.elapsed_time_ms();
                        generation_restore_ms_acc.fetch_add(restore_ms, std::memory_order_relaxed);

                        powerserve::KVCacheInterface *cpu_kv = get_cpu_kv_cache_for_route(context);
                        context.m_model_ptr->kv_cache = private_ggml_kv->kv_cache.get();
                        context.m_model_ptr->ggml_kv_override = private_ggml_kv.get();
                        struct GgmlKvOverrideGuard {
                            powerserve::Model *model = nullptr;
                            powerserve::KVCacheInterface *cpu_kv = nullptr;
                            ~GgmlKvOverrideGuard() {
                                if (model != nullptr) {
                                    model->ggml_kv_override = nullptr;
                                    if (cpu_kv != nullptr) {
                                        model->kv_cache = cpu_kv;
                                    }
                                }
                            }
                        } kv_guard{context.m_model_ptr.get(), cpu_kv};
                        (void)kv_guard;

                        LocalDecodeExecutor decode_executor;
                        generation_candidates[i] = run_blocking_decode_task_from_artifact(
                            decode_executor,
                            input,
                            tokenizer,
                            prefill_results[i],
                            decode_task
                        );
                    } else {
                        LocalDecodeExecutor decode_executor;
                        generation_candidates[i] = run_blocking_decode_task_from_artifact(
                            decode_executor,
                            input,
                            tokenizer,
                            prefill_results[i],
                            decode_task
                        );
                    }

                    prefill_samplers[i].reset();
                    prefill_kv_snapshots[i].reset();
                    server_context.kv_cache_manager->release(input.request_id);
                    generation_decode_ms_acc.fetch_add(decode_timer.elapsed_time_ms(), std::memory_order_relaxed);
                    const long decode_end_ms = static_cast<long>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()
                        ).count()
                    );
                    long previous_decode_end = generation_decode_end_ms.load(std::memory_order_relaxed);
                    while (previous_decode_end < decode_end_ms &&
                           !generation_decode_end_ms.compare_exchange_weak(
                               previous_decode_end,
                               decode_end_ms,
                               std::memory_order_relaxed,
                               std::memory_order_relaxed
                           )) {
                    }
                    POWERSERVE_LOG_DEBUG(
                        "scheduler2 dag decode end: request_id={}, backend={}",
                        input.request_id,
                        BackendRouter::backend_name(decode_route.backend)
                    );
                } catch (...) {
                    prefill_samplers[i].reset();
                    prefill_kv_snapshots[i].reset();
                    server_context.kv_cache_manager->release(input.request_id);
                    throw;
                }
            },
            .debug_name = "generation_decode_" + std::to_string(i),
        });
    }

    dag_nodes.push_back({
        .node_id = generation_merge_node_id,
        .type = powerserve::Scheduler2TaskType::UNKNOWN,
        .request_id = next_rag_request_id(),
        .backend = powerserve::BackendKind::CPU,
        .dependencies = generation_decode_node_ids,
        .fn = [&]() {
            const GenerationMergeResult merge_result = merge_generation_candidates_v1(generation_candidates);
            const GenerationDecodeCandidate &selected_candidate = generation_candidates.at(merge_result.selected_candidate_idx);
            response.answer = selected_candidate.output.m_text;
            response.generation_prefill_queue_wait_ms = 0;
            for (const auto &candidate : generation_candidates) {
                response.generation_prefill_queue_wait_ms += candidate.queue_wait_ms;
            }
            response.decode_task_count = generation_tasks.size();
            response.candidate_count = generation_candidates.size();
            response.selected_answer_source = merge_result.selected_source;
            response.merge_policy_version = merge_result.merge_policy_version;

            response.decode_task_summaries.clear();
            response.decode_task_summaries.reserve(generation_candidates.size());
            for (const auto &candidate : generation_candidates) {
                std::string preview = candidate.output.m_text;
                constexpr size_t kMaxPreviewChars = 200;
                if (preview.size() > kMaxPreviewChars) {
                    preview.resize(kMaxPreviewChars);
                    remove_incomplete_utf8_char(preview);
                }
                response.decode_task_summaries.push_back({
                    .source = candidate.source,
                    .output_tokens = candidate.output.m_output_num_token > 1 ? candidate.output.m_output_num_token - 1 : 0,
                    .output_chars = candidate.output.m_text.size(),
                    .stop_reason = candidate.output.m_stop_reason.value_or("unknown"),
                    .text_preview = std::move(preview),
                });
            }
            segmented_prefill_used = true;
        },
        .debug_name = "generation_merge",
    });

    size_t edge_count = 0;
    for (const auto &node : dag_nodes) {
        edge_count += node.dependencies.size();
    }
    POWERSERVE_LOG_INFO("Scheduler2 submit big dag: nodes={}, edges={}", dag_nodes.size(), edge_count);

    try {
        server_context.scheduler2->submit_dag(std::move(dag_nodes)).get();
    } catch (...) {
        for (size_t i = 0; i < prefill_samplers.size(); ++i) {
            prefill_samplers[i].reset();
            prefill_kv_snapshots[i].reset();
            if (generation_input_ready[i]) {
                server_context.kv_cache_manager->release(generation_inputs[i].request_id);
            }
        }
        if (!selected_context.empty()) {
            const std::string generation_prompt = build_generation_prompt(request.query, selected_context);
            ModelInput fallback_generation_input = make_generation_input(request, generation_prompt);
            apply_generation_route_to_input(fallback_generation_input, generation_route_plan);
            const ModelOutput generation_out = completion(server_context, fallback_generation_input);
            response.answer = generation_out.m_text;
            response.decode_task_count = 1;
            response.selected_answer_source = "original";
            response.candidate_count = 1;
            response.merge_policy_version = "v1-rule";
            response.decode_task_summaries = {
                DecodeTaskDebugSummary{
                    .source = "original",
                    .output_tokens = generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0,
                    .output_chars = generation_out.m_text.size(),
                    .stop_reason = generation_out.m_stop_reason.value_or("unknown"),
                    .text_preview = generation_out.m_text,
                }
            };
            response.generation_decode_steps =
                generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0;
        } else {
            throw;
        }
    }

    response.sub_queries = query_branch.sub_queries;
    response.query_used = request.query;
    response.metrics.indexing_ms = doc_branch.indexing_ms;
    response.metrics.query_expand_ms = query_branch.query_expand_ms;
    response.metrics.query_embedding_ms = query_embedding_ms_acc.load(std::memory_order_relaxed);
    response.metrics.embedding_ms = doc_branch.doc_embedding_ms + response.metrics.query_embedding_ms;
    response.metrics.searching_ms = searching_ms_acc.load(std::memory_order_relaxed);
    response.metrics.reranking_ms = reranking_ms_acc.load(std::memory_order_relaxed);
    response.top_k_indices = std::move(top_k_indices_collected);
    response.top_n_indices = std::move(top_n_indices_collected);

    response.generation_sub_metrics.prefill_sum_ms = generation_prefill_ms_acc.load(std::memory_order_relaxed);
    response.generation_sub_metrics.decode_sum_ms = generation_decode_ms_acc.load(std::memory_order_relaxed);
    const long prefill_begin_ms = generation_prefill_begin_ms.load(std::memory_order_relaxed);
    const long prefill_end_ms = generation_prefill_end_ms.load(std::memory_order_relaxed);
    response.generation_sub_metrics.prefill_ms = (prefill_begin_ms >= 0 && prefill_end_ms >= prefill_begin_ms)
        ? static_cast<size_t>(prefill_end_ms - prefill_begin_ms)
        : 0;
    const long decode_begin_ms = generation_decode_begin_ms.load(std::memory_order_relaxed);
    const long decode_end_ms = generation_decode_end_ms.load(std::memory_order_relaxed);
    response.generation_sub_metrics.decode_ms = (decode_begin_ms >= 0 && decode_end_ms >= decode_begin_ms)
        ? static_cast<size_t>(decode_end_ms - decode_begin_ms)
        : 0;
    response.generation_sub_metrics.bridge_ms = generation_bridge_ms_acc.load(std::memory_order_relaxed);
    response.generation_sub_metrics.kv_snapshot_ms = generation_snapshot_ms_acc.load(std::memory_order_relaxed);
    response.generation_sub_metrics.kv_restore_ms = generation_restore_ms_acc.load(std::memory_order_relaxed);
    response.generation_sub_metrics.kv_snapshot_bytes = generation_snapshot_bytes_acc.load(std::memory_order_relaxed);

    response.generation_segmented_prefill_used = segmented_prefill_used;
    if (segmented_prefill_used && !generation_candidates.empty()) {
        size_t selected_candidate_idx = 0;
        for (size_t i = 0; i < generation_candidates.size(); ++i) {
            if (generation_candidates[i].source == response.selected_answer_source) {
                selected_candidate_idx = i;
                break;
            }
        }
        response.generation_decode_steps =
            generation_candidates[selected_candidate_idx].output.m_output_num_token > 1
            ? generation_candidates[selected_candidate_idx].output.m_output_num_token - 1
            : 0;
    }

    response.context_chunks = std::move(selected_context);
    const long generation_begin_ms = generation_start_ms.load(std::memory_order_relaxed);
    if (generation_begin_ms >= 0) {
        const long now_ms = static_cast<long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
        response.metrics.generation_ms = static_cast<size_t>(std::max<long>(0, now_ms - generation_begin_ms));
    } else {
        response.metrics.generation_ms =
            response.generation_sub_metrics.prefill_ms + response.generation_sub_metrics.decode_ms;
    }
    response.metrics.total_ms = total_timer.elapsed_time_ms();
    log_rag_stage_metrics(response);
    return response;
}
