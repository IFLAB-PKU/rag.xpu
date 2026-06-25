#!/usr/bin/env bash
set -euo pipefail

SERIAL="3B15940035V00000"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DOC="$LOCAL_DIR/rag_long_doc.txt"
LOCAL_PAYLOAD="$LOCAL_DIR/rag_payload_once.json"
PHONE_DOC_DIR="/data/local/tmp/shuhua/doc"
PHONE_DOC="$PHONE_DOC_DIR/rag_long_doc.txt"

PROFILE="${1:-npu_cpu}"
PD_ORCHESTRATOR="${2:-false}"

MODE="hetero_parallel"
GEN_PREFILL_BACKEND="npu"
GEN_DECODE_BACKEND="cpu"

case "$PROFILE" in
  pure_cpu_sequential|cpu_sequential)
    MODE="sequential"
    GEN_PREFILL_BACKEND="cpu"
    GEN_DECODE_BACKEND="cpu"
    ;;
  pure_npu_sequential|npu_sequential)
    MODE="sequential"
    GEN_PREFILL_BACKEND="npu"
    GEN_DECODE_BACKEND="npu"
    ;;
  npu_cpu|hetero_npu_cpu)
    MODE="hetero_parallel"
    GEN_PREFILL_BACKEND="npu"
    GEN_DECODE_BACKEND="cpu"
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage:
  ./tests/run_rag_once.sh [profile] [enable_pd_orchestrator]

Profiles:
  pure_cpu_sequential  prefill=cpu, decode=cpu, mode=sequential
  pure_npu_sequential  prefill=npu, decode=npu, mode=sequential
  npu_cpu              prefill=npu, decode=cpu, mode=hetero_parallel

Enable PD Orchestrator (optional, default false):
  true   use PDOrchestrator for generation (prefill-decode separation)
  false  use SequentialSegmentPrefillQueueDemo (default)

Examples:
  ./tests/run_rag_once.sh pure_cpu_sequential
  ./tests/run_rag_once.sh npu_cpu true
  ./tests/run_rag_once.sh npu_cpu false
EOF
    exit 0
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "try: pure_cpu_sequential | pure_npu_sequential | npu_cpu" >&2
    exit 2
    ;;
esac

if [[ -z "$PD_ORCHESTRATOR" ]]; then
    PD_ORCHESTRATOR="false"
fi

case "$PD_ORCHESTRATOR" in
  true|True|TRUE|1)
    PD_ORCHESTRATOR="true"
    ;;
  false|False|FALSE|0)
    PD_ORCHESTRATOR="false"
    ;;
  *)
    echo "unknown enable_pd_orchestrator: $PD_ORCHESTRATOR" >&2
    echo "try: true | false" >&2
    exit 2
    ;;
esac
if [[ ! -f "$LOCAL_DOC" ]]; then
  echo "missing doc: $LOCAL_DOC" >&2
  exit 1
fi

echo "[1/4] create phone doc dir"
adb -s "$SERIAL" shell "mkdir -p $PHONE_DOC_DIR"

echo "[2/4] push long doc to phone"
adb -s "$SERIAL" push "$LOCAL_DOC" "$PHONE_DOC" >/dev/null

echo "[3/4] build one-shot payload"
echo "profile=$PROFILE mode=$MODE prefill=$GEN_PREFILL_BACKEND decode=$GEN_DECODE_BACKEND pd_orchestrator=$PD_ORCHESTRATOR"
LOCAL_DOC_FOR_PY="$LOCAL_DOC" \
LOCAL_PAYLOAD_FOR_PY="$LOCAL_PAYLOAD" \
RAG_MODE="$MODE" \
RAG_PREFILL_BACKEND="$GEN_PREFILL_BACKEND" \
RAG_DECODE_BACKEND="$GEN_DECODE_BACKEND" \
RAG_PD_ORCHESTRATOR="$PD_ORCHESTRATOR" \
python3 - <<'PY'
import json
import os
from pathlib import Path

local_doc = os.environ['LOCAL_DOC_FOR_PY']
local_payload = os.environ['LOCAL_PAYLOAD_FOR_PY']
rag_mode = os.environ['RAG_MODE']
rag_prefill_backend = os.environ['RAG_PREFILL_BACKEND']
rag_decode_backend = os.environ['RAG_DECODE_BACKEND']
rag_pd_orchestrator = os.environ['RAG_PD_ORCHESTRATOR'].lower() == 'true'

payload = {
    'doc': Path(local_doc).read_text(encoding='utf-8', errors='ignore'),
    'query': 'OpenAI的发展中体现了哪些取舍？',
  'mode': rag_mode,
  'generation_prefill_backend': rag_prefill_backend,
  'generation_decode_backend': rag_decode_backend,
    'generation_model': 'qwen3-0.6b-base',
    'embedding_model': 'qwen3-embedding-0.6b',
    'rerank_model': 'qwen3-reranker-0.6b',
    'enable_query_expansion': True,
    'enable_pd_orchestrator': rag_pd_orchestrator,
    'top_k': 20,
    'top_n': 5,
    'generation_decode_steps': 192,
    'max_tokens': 192,
    'temperature': 0.1,
}
Path(local_payload).write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding='utf-8',
)
print('payload ready')
PY

echo "[4/4] run one request"
adb -s "$SERIAL" forward tcp:18080 tcp:8080 >/dev/null
curl -sS -X POST http://127.0.0.1:18080/v1/rag \
  -H "Content-Type: application/json" \
  --data-binary @"$LOCAL_PAYLOAD"

echo

echo "done"
echo "phone doc path: $PHONE_DOC"
echo "local payload:  $LOCAL_PAYLOAD"
echo "pd_orchestrator: $PD_ORCHESTRATOR"

# How to run
# ./tests/run_rag_once.sh pure_cpu_sequential
# ./tests/run_rag_once.sh pure_npu_sequential
# ./tests/run_rag_once.sh npu_cpu
# ./tests/run_rag_once.sh npu_cpu true
# ./tests/run_rag_once.sh npu_cpu false