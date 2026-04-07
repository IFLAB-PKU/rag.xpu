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

struct RagResponse {
    std::string answer;
    std::string mode_requested;
    std::string mode_used;
    std::string query_used;
    std::vector<std::string> context_chunks;
    std::vector<size_t> top_k_indices;
    std::vector<size_t> top_n_indices;
    RagStageMetrics metrics;
};

inline size_t next_rag_request_id() {
    static std::atomic_size_t request_counter{0};
    return request_counter.fetch_add(1, std::memory_order_relaxed);
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

inline std::string maybe_expand_rag_query(
    ServerContext &server_context,
    const RagRequest &request,
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
        "Rewrite the user query into one short retrieval query. "
        "Do not answer the question.\n"
        "Query: " + request.query + "\n"
        "Rewrite:";

    ModelInput expand_input = make_generation_input(request, expand_prompt);
    expand_input.m_model = model_for_expand;
    expand_input.m_temperature = 0.01F;
    expand_input.m_max_num_token = std::min<size_t>(request.max_tokens, size_t{32});

    const ModelOutput expand_output = completion(server_context, expand_input);
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
    const std::string expanded_query = maybe_expand_rag_query(server_context, request, response.metrics.query_expand_ms);
    response.query_used = expanded_query;

    // 4) Query embedding
    stage_timer = Timer{};
    const ModelOutput query_embedding_out = embedding(server_context, make_embedding_input(request.embedding_model, expanded_query));
    if (query_embedding_out.m_embedding.empty()) {
        throw std::runtime_error("query embedding is empty");
    }
    response.metrics.query_embedding_ms = stage_timer.elapsed_time_ms();
    response.metrics.embedding_ms += response.metrics.query_embedding_ms;

    // 5) Searching (FAISS IndexFlatIP)
    stage_timer = Timer{};
    const std::vector<size_t> faiss_top_indices = rag_search_faiss_ip(
        doc_embeddings,
        doc_embedding_source_indices,
        query_embedding_out.m_embedding,
        request.top_k
    );

    const size_t actual_top_k = faiss_top_indices.size();
    std::vector<std::string> top_k_docs;
    top_k_docs.reserve(actual_top_k);
    for (size_t i = 0; i < actual_top_k; ++i) {
        response.top_k_indices.push_back(faiss_top_indices[i]);
        top_k_docs.push_back(chunks[faiss_top_indices[i]]);
    }
    response.metrics.searching_ms = stage_timer.elapsed_time_ms();
    if (top_k_docs.empty()) {
        throw std::runtime_error("retrieval returns empty top_k docs");
    }

    // 6) Reranking
    stage_timer = Timer{};
    const ModelOutput rerank_out = rerank(server_context, make_rerank_input(request, expanded_query, top_k_docs));

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
    const std::string generation_prompt = build_generation_prompt(expanded_query, selected_context);
    const ModelOutput generation_out = completion(server_context, make_generation_input(request, generation_prompt));
    response.answer = generation_out.m_text;
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
        size_t query_expand_ms = 0;
    };

    RagResponse response;
    response.mode_requested = request.mode;
    response.mode_used = "hetero_parallel";
    response.query_used = request.query;

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
        out.expanded_query = maybe_expand_rag_query(server_context, request, out.query_expand_ms);
        return out;
    });

    const QueryBranchOutput query_branch = query_branch_future.get();
    response.query_used = query_branch.expanded_query;

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
    const std::string generation_prompt = build_generation_prompt(query_branch.expanded_query, selected_context);
    const ModelOutput generation_out = completion(server_context, make_generation_input(request, generation_prompt));
    response.answer = generation_out.m_text;
    response.context_chunks = std::move(selected_context);
    response.metrics.generation_ms = stage_timer.elapsed_time_ms();

    response.metrics.total_ms = total_timer.elapsed_time_ms();
    return response;
}
