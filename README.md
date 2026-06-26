# HybridRAG

A heterogeneous RAG inference system for mobile NPU/CPU devices, built on the Qwen3 model family. It runs an end-to-end retrieval-augmented generation pipeline with NPU prefill and CPU decode orchestration, targeting Qualcomm Snapdragon platforms via QNN.

> This project evolves from the [PowerServe](https://github.com/powerserve-project/PowerServe) LLM serving framework. The original README has been superseded by this document.

---

## Table of Contents

1. [Features](#features)
2. [Supported Models](#supported-models)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Prerequisites](#prerequisites)
6. [Model Preparation](#model-preparation)
7. [Build](#build)
8. [Deploy](#deploy)
9. [How to Test](#how-to-test)
10. [API Examples](#api-examples)
11. [Known Issues](#known-issues)

---

## Features

- **End-to-end RAG pipeline** with 6 stages: indexing, query expansion, query embedding, searching, reranking, and generation.
- **Heterogeneous execution**: NPU prefill + CPU decode for generation, CPU-based embedding/reranking/FAISS retrieval.
- **Qwen3 model family support**: base, embedding, reranker, and 4B generation models.
- **Prefill/Decode (PD) orchestration** with KV snapshot isolation, NPU-to-CPU KV bridge, and private KV decode.
- **scheduler2 DAG scheduler**: full RAG pipeline expressed as a dependency graph, with FIFO or critical-score scheduling modes.
- **OpenAI-compatible endpoints**: `/completions`, `/chat/completions`, `/embeddings`, `/rerank`, `/rag`.

---

## Supported Models

| Model | Role | Backend |
|-------|------|---------|
| `qwen3-0.6b-base` | Generation / Query expansion | CPU / NPU |
| `qwen3-4b` | Main generation model | CPU / NPU |
| `qwen3-embedding-0.6b` | Document/query embedding | CPU / NPU(experimental) |
| `qwen3-reranker-0.6b` | Reranking | CPU / NPU |

In the default RAG pipeline:

- **Indexing** and **query embedding** use `qwen3-embedding-0.6b`.
- **Query expansion** (optional) uses `qwen3-0.6b-base`, which is unloaded after expansion to avoid NPU shared-memory conflicts.
- **Reranking** uses `qwen3-reranker-0.6b`.
- **Generation** uses `qwen3-4b` (or `qwen3-0.6b-base` if configured).

---

## Architecture

### RAG Pipeline

```
Document  →  Indexing  →  Document Embeddings
                                        ↓
Query → Query Expansion → Query Embedding → Searching (FAISS) → Reranking → Generation → Answer
```

The 6 stages are:

1. **Indexing**: Split the document into chunks and encode them with the embedding model.
2. **Query Expansion** (optional): Rewrite the user query into several sub-queries.
3. **Query Embedding**: Encode the original query and sub-queries.
4. **Searching**: Retrieve top-K chunks per query with FAISS `IndexFlatIP`, then merge and deduplicate.
5. **Reranking**: Score query-chunk pairs and return top-N context chunks.
6. **Generation**: Produce one or more answer candidates using the selected context, then merge them.

### Execution Modes

The `/rag` endpoint selects the execution mode via the `mode` field:

| Mode | Scheduler | Generation Prefill | Generation Decode |
|------|-----------|-------------------|-------------------|
| `sequential` | Single backend, serial | `cpu` or `npu` | Same as prefill |
| `carrier_baseline` | `std::async` parallelism + serial generation | `npu` | `cpu` |
| `hetero_parallel` | scheduler2 DAG, FIFO + work stealing | `npu` | `cpu` |
| `npu_cpu_cs` | scheduler2 DAG + critical-score scheduling | `npu` | `cpu` |

The last two modes are the focus of the project. They use NPU for high-throughput prefill and CPU for low-latency decode, while the scheduler2 DAG expresses the whole pipeline as dependencies.

### Prefill / Decode Orchestration

- `PDOrchestrator` dispatches prefill and decode tasks to dedicated workers.
- `set_generation_backend_route()` switches the active KV cache between CPU (GGML) and NPU (QNN).
- After NPU prefill, `sync_qnn_kv_to_cpu()` copies the KV state to CPU memory.
- `GGMLKV::save_snapshot/restore_snapshot` captures the CPU KV cache. In scheduler2 mode, a **base+delta** snapshot is used: the shared prefix is saved once and each candidate restores only its delta, lowering memory and restore cost.
- CPU decode uses a **private KV** via a thread-local `ModelExecutionRoute`, so it does not corrupt shared model state while the next NPU prefill overlaps.

### scheduler2

`scheduler2` (`src/scheduler2/`) is a CPU+NPU DAG executor:

- Tasks are submitted as `Scheduler2DagNode`s with explicit dependencies.
- Two workers consume a CPU queue and an NPU queue.
- By default, workers can steal non-generation tasks across backends to maximize utilization.
- In `npu_cpu_cs` mode, tasks are ordered by a profile-derived critical score and generation tasks are pinned to their target backend.

---

## Directory Structure

```
HybridRAG/
├── app/
│   └── server/            # HTTP server, RAG pipeline, OpenAI-compatible handlers
├── src/
│   ├── backend/           # GGML and QNN backends
│   ├── model/qwen3/       # Qwen3 model implementation
│   ├── scheduler2/        # DAG scheduler, backend router, KV cache manager
│   └── ...
├── tests/                 # Test scripts and workload documents
│   ├── run_server.sh
│   ├── run_rag_once.sh
│   ├── run_rag_workload.sh
│   ├── run_rag_benchmark.sh
│   └── workloads/
├── tools/
│   ├── gguf_export.py     # CPU model conversion
│   └── qnn_converter/     # NPU model conversion
├── docs/                  # Design and phase documents
├── README.md
└── requirements.txt
```

---

## Prerequisites

- Linux host with Docker (recommended)
- Android NDK
- Qualcomm QNN SDK
- `adb` and an Android device with USB debugging
- Python dependencies:

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

Set the usual environment variables inside your build container:

```bash
export ANDROID_NDK=<path-to-ndk>
export QNN_SDK=<path-to-qnn>
source $QNN_SDK/bin/envsetup.sh
```

---

## Model Preparation

### 1. Download Hugging Face Models

Download the Qwen3 checkpoints you need, for example:

- `Qwen/Qwen3-0.6B-Base`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-Embedding-0.6B`
- `Qwen/Qwen3-Reranker-0.6B`

### 2. Convert CPU Model (GGUF)

```bash
python ./tools/gguf_export.py \
    -m <path-to-hf-model> \
    -o models/<model-name>
```

### 3. Convert NPU Model (QNN)

Follow `tools/qnn_converter/README.md`. A typical flow is:

```bash
cd tools/qnn_converter

python converter.py \
    --model-folder <path-to-hf-model> \
    --model-name <model-name> \
    --system-prompt-file <system-prompt-file> \
    --prompt-file <prompt-file> \
    --batch-sizes 1 128 \
    --artifact-name <artifact-name> \
    --n-model-chunk <n> \
    --output-folder ./<output> \
    --build-folder ./<tmp> \
    --silent \
    --clear-build-files \
    --soc <soc-number>
```

Then merge the QNN artifacts into the GGUF model:

```bash
python ./tools/gguf_export.py \
    -m <path-to-hf-model> \
    --qnn-path tools/qnn_converter/<output> \
    -o ./<model-name>-model
```

### 4. Create PowerServe Workspace

```bash
mkdir -p models
./powerserve create \
    -m ./<model-name>-model \
    --exe-path ./build/out \
    -o ./models/<model-name>
```

Repeat for each model. The final on-device layout should look like:

```
models/
├── bin/
│   └── powerserve-server
├── lib/
├── qnn_libs/
├── qwen3-0.6b-base/
├── qwen3-4b/
├── qwen3-embedding-0.6b/
└── qwen3-reranker-0.6b/
```

---

## Build

### Docker Build Environment

Use your own Android/QNN cross-compilation image. For example:

```bash
docker run --rm -it \
    -v "$(pwd):/workspace" \
    -v "<path-to-models>:/models" \
    -w /workspace \
    <your-android-qnn-build-image> /bin/bash
```

Inside the container, set the environment variables above, then run the CMake commands below.

### Android aarch64 with QNN

Inside the Docker container:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-35 \
    -DGGML_OPENMP=OFF \
    -DPOWERSERVE_WITH_QNN=ON \
    -DPOWERSERVE_ENABLE_HTPRPCPOLL=ON \
    -DPOWERSERVE_ENABLE_HMXPWRCFG=ON \
    -DPOWERSERVE_USE_DUMMY=ON

cmake --build build
```

The server binary is `build/out/powerserve-server`.

### CPU-only Server (optional)

Some test scripts expect a CPU-only binary named `powerserve-server-cpu`. Build it with QNN disabled:

```bash
cmake -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-35 \
    -DGGML_OPENMP=OFF \
    -DPOWERSERVE_WITH_QNN=OFF \
    -DPOWERSERVE_USE_DUMMY=ON

cmake --build build-cpu
cp build-cpu/out/powerserve-server build/out/powerserve-server-cpu
```

### Build Notes

- The project links against an Android-format `libfaiss.so`, so **local x86 builds will fail at link time**. Always cross-compile for Android aarch64.
- On some platforms (e.g., SA8295 / Hexagon V68), disable HTP RPC polling and HMX power config:
  ```bash
  -DPOWERSERVE_ENABLE_HTPRPCPOLL=OFF
  -DPOWERSERVE_ENABLE_HMXPWRCFG=OFF
  ```
- `POWERSERVE_USE_DUMMY=ON` is recommended for SM8650/8750-class devices.

---

## Deploy

Push the server binary to the device:

```bash
adb push build/out/powerserve-server /data/local/tmp/<project>/models/bin/
```

Start the server with the helper script:

```bash
# NPU mode
./tests/run_server.sh npu

# CPU-only mode (requires powerserve-server-cpu)
./tests/run_server.sh cpu
```

Or start it manually for full logs:

```bash
adb shell
export LD_LIBRARY_PATH=/data/local/tmp/<project>/models/lib:/data/local/tmp/<project>/models/qnn_libs
cd /data/local/tmp/<project>
./models/bin/powerserve-server -d ./models --port 8080
```

---

## How to Test

### Single RAG Request

```bash
./tests/run_rag_once.sh npu_cpu
```

Supported profiles:

| Profile | Mode | Prefill | Decode |
|---------|------|---------|--------|
| `pure_cpu_sequential` | `sequential` | cpu | cpu |
| `pure_npu_sequential` | `sequential` | npu | npu |
| `npu_cpu` | `hetero_parallel` | npu | cpu |
| `npu_cpu_cs` | `npu_cpu_cs` | npu | cpu |
| `carrier_baseline` | `carrier_baseline` | npu | cpu |

### Workload Tests

```bash
./tests/run_rag_workload.sh 4K npu_cpu
./tests/run_rag_workload.sh 8K npu_cpu_cs
```

Available workloads: `4K`, `6K`, `8K`, `10K`.

### Benchmark

```bash
./tests/run_rag_benchmark.sh npu_cpu
./tests/run_rag_benchmark.sh npu_cpu_cs
```

This runs all workloads and prints a per-stage timing table.

### Environment Variable Overrides

```bash
RAG_EXPANSION_MODEL=qwen3-0.6b-base \
RAG_GENERATION_MODEL=qwen3-4b \
    ./tests/run_rag_workload.sh 4K npu_cpu

RAG_TOP_N=10 \
RAG_CANDIDATE_REPEATS=3 \
    ./tests/run_rag_workload.sh 8K npu_cpu_cs
```

| Variable | Purpose |
|----------|---------|
| `RAG_EXPANSION_MODEL` | Query expansion model |
| `RAG_GENERATION_MODEL` | Answer generation model |
| `RAG_TOP_N` | Number of chunks after reranking |
| `RAG_CANDIDATE_REPEATS` | Repeat count per sub-query candidate |

---

## API Examples

Forward the device port:

```bash
adb forward tcp:18080 tcp:8080
```

### RAG

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/rag \
  -H "Content-Type: application/json" \
  -d '{
    "doc": "OpenAI was founded in 2015 ...",
    "query": "What trade-offs has OpenAI made?",
    "mode": "hetero_parallel",
    "generation_prefill_backend": "npu",
    "generation_decode_backend": "cpu",
    "generation_model": "qwen3-4b",
    "embedding_model": "qwen3-embedding-0.6b",
    "rerank_model": "qwen3-reranker-0.6b",
    "expansion_model": "qwen3-0.6b-base",
    "enable_query_expansion": true,
    "top_k": 20,
    "top_n": 5,
    "generation_decode_steps": 64,
    "max_tokens": 64,
    "temperature": 0.1
  }'
```

Key response fields:

```json
{
  "answer": "...",
  "mode_requested": "hetero_parallel",
  "mode_used": "hetero_parallel",
  "stage_metrics_ms": {
    "indexing": 5086,
    "query_expand": 1115,
    "query_embedding": 1242,
    "embedding": 6054,
    "searching": 16,
    "reranking": 4401,
    "generation": 26973,
    "total": 30423
  },
  "debug": {
    "generation_prefill_backend_target": "npu",
    "generation_decode_backend_target": "cpu",
    "generation_kv_bridge_available": true,
    "generation_segmented_prefill_used": true,
    "generation_sub_metrics": {
      "prefill_ms": 2283,
      "decode_ms": 26179,
      "bridge_ms": 0,
      "kv_snapshot_ms": 120,
      "kv_restore_ms": 45,
      "kv_snapshot_bytes": 36700160
    }
  }
}
```

### Completions

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-base",
    "prompt": "Briefly introduce OpenAI.",
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

### Chat Completions

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-base",
    "messages": [
      {"role": "user", "content": "Briefly introduce OpenAI."}
    ],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

### Embeddings

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-0.6b",
    "input": "When was OpenAI founded?"
  }'
```

### Rerank

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-reranker-0.6b",
    "query": "When was OpenAI founded?",
    "documents": [
      "OpenAI was founded in 2015.",
      "GPT-4 was released in 2023."
    ],
    "top_n": 2
  }'
```

---

## Known Issues
- **Embedding NPU hybrid path is experimental**: Qwen3 embedding supports a QNN backbone + CPU finalization path, but numerical precision issues may still exist.

