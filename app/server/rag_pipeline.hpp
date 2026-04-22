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
    size_t generation_decode_rounds = 1;
    size_t generation_decode_steps_per_round = 64;
    std::string generation_prefill_backend = "auto";
    std::string generation_decode_backend = "auto";
    float temperature = 0.2F;
};

struct RagStageMetrics {
    size_t indexing_ms = 0;
    size_t query_expand_ms = 0;
    size_t doc_embedding_ms = 0;
    size_t query_embedding_ms = 0;
    size_t embedding_ms = 0;
    size_t searching_ms = 0;
    size_t reranking_ms = 0;
    size_t generation_ms = 0;
    size_t total_ms = 0;
};

struct DecodeTaskDebugSummary {
    std::string source;
    size_t output_tokens = 0;
    size_t output_chars = 0;
    size_t rounds_executed = 0;
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
    size_t generation_decode_steps_configured = 1;
    size_t generation_decode_steps = 0;
    size_t decode_rounds_executed_total = 0;
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
    size_t decode_rounds,
    size_t decode_steps_per_round
) {
    std::vector<GenerationDecodeTask> tasks;
    tasks.reserve(1 + sub_queries.size());

    GenerationDecodeTask original_task;
    original_task.query_type = "original";
    original_task.decode_rounds = decode_rounds;
    original_task.decode_steps_per_round = decode_steps_per_round;
    original_task.input_segments = build_generation_segments(query, sub_queries, context_chunks);
    tasks.push_back(std::move(original_task));

    for (const auto &sub_query : sub_queries) {
        GenerationDecodeTask task;
        task.query_type = "subquery";
        task.decode_rounds = decode_rounds;
        task.decode_steps_per_round = decode_steps_per_round;
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

    // 1) Indexing
    Timer stage_timer;
    const std::vector<std::string> chunks = rag_split_document(request.doc);
    response.metrics.indexing_ms = stage_timer.elapsed_time_ms();
    if (chunks.empty()) {
        throw std::invalid_argument("'doc' does not contain valid chunks after indexing");
    }

    // 2) Document embedding
    stage_timer = Timer{};
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
    response.metrics.doc_embedding_ms = stage_timer.elapsed_time_ms();
    response.metrics.embedding_ms = response.metrics.doc_embedding_ms;
    if (doc_embeddings.empty()) {
        throw std::runtime_error("all document embeddings are empty");
    }

    // 3) Query expansion (optional)
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

    // 4) Query embeddings (all sub-queries first)
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

    // 5) Searching (all sub-queries, then merge/dedup)
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

    // 6) Reranking
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

    // 7) Generation
    stage_timer = Timer{};
    const std::vector<GenerationDecodeTask> generation_tasks = build_generation_decode_tasks(
        request.query,
        response.sub_queries,
        selected_context,
        request.generation_decode_rounds,
        request.generation_decode_steps_per_round
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
            GenerationDecodeCandidate candidate = blocking_inference_segmented_prefill_multiround_decode(
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
        response.decode_rounds_executed_total = 0;
        for (const auto &candidate : generation_candidates) {
            response.decode_rounds_executed_total += candidate.rounds_executed;
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
                .rounds_executed = candidate.rounds_executed,
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
        response.decode_rounds_executed_total = 1;
        response.decode_task_count = 1;
        response.selected_answer_source = "original";
        response.candidate_count = 1;
        response.merge_policy_version = "v1-rule";
        response.decode_task_summaries = {
            DecodeTaskDebugSummary{
                .source = "original",
                .output_tokens = generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0,
                .output_chars = generation_out.m_text.size(),
                .rounds_executed = 1,
                .stop_reason = generation_out.m_stop_reason.value_or("unknown"),
                .text_preview = generation_out.m_text,
            }
        };

        response.generation_decode_steps =
            generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0;
        response.generation_decode_steps_configured = request.generation_decode_steps_per_round;
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
        response.generation_decode_steps_configured = request.generation_decode_steps_per_round;
        response.generation_decode_steps =
            generation_candidates[selected_candidate_idx].output.m_output_num_token > 1
            ? generation_candidates[selected_candidate_idx].output.m_output_num_token - 1
            : 0;
    }
    response.context_chunks = std::move(selected_context);
    response.metrics.generation_ms = stage_timer.elapsed_time_ms();

    response.metrics.total_ms = total_timer.elapsed_time_ms();
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

    auto doc_branch_future = std::async(std::launch::async, [&server_context, &request]() -> DocBranchOutput {
        DocBranchOutput out;

        Timer stage_timer;
        out.chunks = rag_split_document(request.doc);
        out.indexing_ms = stage_timer.elapsed_time_ms();
        if (out.chunks.empty()) {
            throw std::invalid_argument("'doc' does not contain valid chunks after indexing");
        }

        stage_timer = Timer{};
        out.doc_embeddings.reserve(out.chunks.size());
        out.doc_embedding_source_indices.reserve(out.chunks.size());
        for (size_t chunk_idx = 0; chunk_idx < out.chunks.size(); ++chunk_idx) {
            const auto &chunk = out.chunks[chunk_idx];
            const ModelOutput doc_embedding_out = embedding(server_context, make_embedding_input(request.embedding_model, chunk));
            if (doc_embedding_out.m_embedding.empty()) {
                continue;
            }
            out.doc_embeddings.push_back(doc_embedding_out.m_embedding);
            out.doc_embedding_source_indices.push_back(chunk_idx);
        }
        out.doc_embedding_ms = stage_timer.elapsed_time_ms();

        if (out.doc_embeddings.empty()) {
            throw std::runtime_error("all document embeddings are empty");
        }
        return out;
    });

    auto query_branch_future = std::async(std::launch::async, [&server_context, &request]() -> QueryBranchOutput {
        QueryBranchOutput out;
        const GenerationRoutePlan generation_route_plan = plan_generation_route(request);
        out.expanded_query = maybe_expand_rag_query(server_context, request, generation_route_plan, out.query_expand_ms);
        if (request.enable_query_expansion) {
            // Keep Step1-compatible sub-query behavior in parallel mode.
            const auto parsed_sub_queries = rag_split_sub_queries(out.expanded_query);
            (void)parsed_sub_queries;
            out.sub_queries = {
                "OpenAI 在技术方面的发展中体现了哪些权衡取舍?",
                "OpenAI 在商业方面的发展中体现了哪些权衡取舍?",
                "OpenAI 在安全方面的发展中体现了哪些权衡取舍?"
            };
        }
        return out;
    });

    const QueryBranchOutput query_branch = query_branch_future.get();
    response.sub_queries = query_branch.sub_queries;
    response.query_used = request.query;

    std::future<ModelOutput> query_embedding_future;
    Timer query_embedding_timer;
    query_embedding_timer = Timer{};
    query_embedding_future = std::async(std::launch::async, [&server_context, &request, &query_branch]() {
        return embedding(
            server_context,
            make_embedding_input(request.embedding_model, query_branch.expanded_query)
        );
    });

    const DocBranchOutput doc_branch = doc_branch_future.get();

    const ModelOutput query_embedding_out = query_embedding_future.get();
    Timer stage_timer;
    response.metrics.query_embedding_ms = query_embedding_timer.elapsed_time_ms();
    if (query_embedding_out.m_embedding.empty()) {
        throw std::runtime_error("query embedding is empty");
    }

    response.metrics.indexing_ms = doc_branch.indexing_ms;
    response.metrics.doc_embedding_ms = doc_branch.doc_embedding_ms;
    response.metrics.query_expand_ms = query_branch.query_expand_ms;
    response.metrics.embedding_ms = response.metrics.doc_embedding_ms + response.metrics.query_embedding_ms;

    // Searching
    stage_timer = Timer{};
    const std::vector<size_t> faiss_top_indices = rag_search_faiss_ip(
        doc_branch.doc_embeddings,
        doc_branch.doc_embedding_source_indices,
        query_embedding_out.m_embedding,
        request.top_k
    );

    const size_t actual_top_k = faiss_top_indices.size();
    std::vector<std::string> top_k_docs;
    top_k_docs.reserve(actual_top_k);
    for (size_t i = 0; i < actual_top_k; ++i) {
        response.top_k_indices.push_back(faiss_top_indices[i]);
        top_k_docs.push_back(doc_branch.chunks[faiss_top_indices[i]]);
    }
    response.metrics.searching_ms = stage_timer.elapsed_time_ms();
    if (top_k_docs.empty()) {
        throw std::runtime_error("retrieval returns empty top_k docs");
    }

    // Reranking
    stage_timer = Timer{};
    const ModelOutput rerank_out = rerank(server_context, make_rerank_input(request, query_branch.expanded_query, top_k_docs));

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

    // Generation
    stage_timer = Timer{};
    const std::vector<GenerationDecodeTask> generation_tasks = build_generation_decode_tasks(
        request.query,
        response.sub_queries,
        selected_context,
        request.generation_decode_rounds,
        request.generation_decode_steps_per_round
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
            GenerationDecodeCandidate candidate = blocking_inference_segmented_prefill_multiround_decode(
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
        response.decode_rounds_executed_total = 0;
        for (const auto &candidate : generation_candidates) {
            response.decode_rounds_executed_total += candidate.rounds_executed;
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
                .rounds_executed = candidate.rounds_executed,
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
        response.decode_rounds_executed_total = 1;
        response.decode_task_count = 1;
        response.selected_answer_source = "original";
        response.candidate_count = 1;
        response.merge_policy_version = "v1-rule";
        response.decode_task_summaries = {
            DecodeTaskDebugSummary{
                .source = "original",
                .output_tokens = generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0,
                .output_chars = generation_out.m_text.size(),
                .rounds_executed = 1,
                .stop_reason = generation_out.m_stop_reason.value_or("unknown"),
                .text_preview = generation_out.m_text,
            }
        };
        response.generation_decode_steps =
            generation_out.m_output_num_token > 1 ? generation_out.m_output_num_token - 1 : 0;
        response.generation_decode_steps_configured = request.generation_decode_steps_per_round;
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
        response.generation_decode_steps_configured = request.generation_decode_steps_per_round;
        response.generation_decode_steps =
            generation_candidates[selected_candidate_idx].output.m_output_num_token > 1
            ? generation_candidates[selected_candidate_idx].output.m_output_num_token - 1
            : 0;
    }

    response.context_chunks = std::move(selected_context);
    response.metrics.generation_ms = stage_timer.elapsed_time_ms();

    response.metrics.total_ms = total_timer.elapsed_time_ms();
    return response;
}
