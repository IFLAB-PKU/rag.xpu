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

#include "openai_api.hpp"
#include "rag_pipeline.hpp"

#include "nlohmann/json.hpp"

#include <string>

inline RagRequest parse_rag_request(const nlohmann::json &request) {
    RagRequest rag_request;

    rag_request.doc = request.value("doc", "");
    rag_request.query = request.value("query", "");

    rag_request.mode = request.value("mode", std::string{"sequential"});
    rag_request.enable_query_expansion = request.value("enable_query_expansion", false);

    rag_request.top_k = request.value("top_k", size_t{20});
    rag_request.top_n = request.value("top_n", size_t{5});
    rag_request.max_tokens = request.value("max_tokens", size_t{128});
    rag_request.temperature = request.value("temperature", 0.2F);

    rag_request.generation_model = request.value("generation_model", request.value("model", std::string{}));
    rag_request.embedding_model = request.value("embedding_model", std::string{});
    rag_request.rerank_model = request.value("rerank_model", std::string{});
    rag_request.expansion_model = request.value("expansion_model", std::string{});

    if (rag_request.mode != "sequential" && rag_request.mode != "hetero_parallel") {
        throw std::invalid_argument("'mode' must be one of: sequential, hetero_parallel");
    }
    if (rag_request.top_n == 0 || rag_request.top_k == 0) {
        throw std::invalid_argument("'top_k' and 'top_n' must be > 0");
    }
    if (rag_request.top_n > rag_request.top_k) {
        throw std::invalid_argument("'top_n' must be <= 'top_k'");
    }

    return rag_request;
}

inline nlohmann::json dump_rag_response(const RagResponse &response) {
    const nlohmann::json stage_timeline = nlohmann::json::array({
        nlohmann::json{{"stage", "indexing"}, {"ms", response.metrics.indexing_ms}},
        nlohmann::json{{"stage", "doc_embedding"}, {"ms", response.metrics.doc_embedding_ms}},
        nlohmann::json{{"stage", "query_expand"}, {"ms", response.metrics.query_expand_ms}},
        nlohmann::json{{"stage", "query_embedding"}, {"ms", response.metrics.query_embedding_ms}},
        nlohmann::json{{"stage", "searching"}, {"ms", response.metrics.searching_ms}},
        nlohmann::json{{"stage", "reranking"}, {"ms", response.metrics.reranking_ms}},
        nlohmann::json{{"stage", "generation"}, {"ms", response.metrics.generation_ms}},
        nlohmann::json{{"stage", "total"}, {"ms", response.metrics.total_ms}}
    });

    return {
        {"object", "rag.result"},
        {"mode_requested", response.mode_requested},
        {"mode_used", response.mode_used},
        {"query_used", response.query_used},
        {"answer", response.answer},
        {"context_chunks", response.context_chunks},
        {"debug",
         {{"top_k_indices", response.top_k_indices},
          {"top_n_indices", response.top_n_indices}}},
        {"stage_metrics_ms",
         {{"indexing", response.metrics.indexing_ms},
          {"query_expand", response.metrics.query_expand_ms},
                    {"doc_embedding", response.metrics.doc_embedding_ms},
                    {"query_embedding", response.metrics.query_embedding_ms},
          {"embedding", response.metrics.embedding_ms},
          {"searching", response.metrics.searching_ms},
          {"reranking", response.metrics.reranking_ms},
          {"generation", response.metrics.generation_ms},
                    {"total", response.metrics.total_ms}}},
                {"stage_timeline", stage_timeline}
    };
}

template <class T_Request = httplib::Request, class T_Response = httplib::Response>
inline void handler_rag(ServerContext &server_context, const T_Request &request, T_Response &response) {
    POWERSERVE_LOG_INFO("process rag task");

    RagRequest rag_request;
    try {
        const auto body = nlohmann::json::parse(request.body);
        rag_request = parse_rag_request(body);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
        return;
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
        return;
    }

    try {
        const RagResponse rag_response = run_rag_sequential(server_context, rag_request);
        response_normal(dump_rag_response(rag_response), response);
        POWERSERVE_LOG_INFO("after rag: {}", powerserve::perf_get_mem_result());
    } catch (const std::invalid_argument &err) {
        response_error(err.what(), ErrorType::InvalidRequest, response);
    } catch (const std::exception &err) {
        response_error(err.what(), ErrorType::Server, response);
    } catch (...) {
        response_error("unknown error", ErrorType::Server, response);
    }
}
