#!/usr/bin/env bash
set -euo pipefail

DEVICE_PROFILE_RAW=""
if [[ "$#" -gt 0 ]]; then
    DEVICE_PROFILE_RAW="${!#}"
fi
DEVICE_PROFILE="8gen4"
if [[ "$#" -gt 0 && ( "$DEVICE_PROFILE_RAW" == "8gen4" || "$DEVICE_PROFILE_RAW" == "8gen5" ) ]]; then
    DEVICE_PROFILE="$DEVICE_PROFILE_RAW"
    if [[ "$#" -eq 1 ]]; then
        set --
    else
        set -- "${@:1:$(($# - 1))}"
    fi
fi

case "$DEVICE_PROFILE" in
  8gen4)
    SERIAL="3B15940035V00000"
    ;;
  8gen5)
    SERIAL="3B15CR0014H00000"
    ;;
  *)
    echo "unknown device flag: $DEVICE_PROFILE" >&2
    echo "try: 8gen4 | 8gen5" >&2
    exit 2
    ;;
esac

# Run a single workload (4K/6K/8K) with specified profile.
# Usage:
#   ./tests/run_rag_workload.sh <workload> [profile] [8gen4|8gen5]
#
# Examples:
#   ./tests/run_rag_workload.sh 4K npu_cpu 8gen5
#   ./tests/run_rag_workload.sh 8K pure_cpu_sequential 8gen4
#   ./tests/run_rag_workload.sh 6K pure_npu_sequential 8gen5

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOADS_JSON="$LOCAL_DIR/workloads/workloads.json"
PHONE_DOC_DIR="/data/local/tmp/shuhua/doc"
LOCAL_TMP_DIR="$LOCAL_DIR/.tmp"

WORKLOAD="${1:-}"
PROFILE="${2:-npu_cpu}"

# Validate workload
if [[ -z "$WORKLOAD" ]]; then
echo "Usage: $0 <workload> [profile] [8gen4|8gen5]" >&2
    echo "  workload: 4K | 6K | 8K" >&2
    exit 1
fi

# Resolve payload file and doc file from workloads.json
PAYLOAD_FILE=""
DOC_FILE=""
read -r PAYLOAD_FILE DOC_FILE < <(python3 -c "
import json, sys
with open('$WORKLOADS_JSON') as f:
    data = json.load(f)
if '$WORKLOAD' not in data:
    sys.exit(1)
entry = data['$WORKLOAD']
print(entry.get('payload_file', ''), entry.get('doc_file', ''))
")

if [[ -z "$PAYLOAD_FILE" || -z "$DOC_FILE" ]]; then
    echo "unknown workload: $WORKLOAD" >&2
    echo "available: 4K | 6K | 8K" >&2
    exit 2
fi

# Resolve paths (relative to project root)
PROJECT_ROOT="$(cd "$LOCAL_DIR/.." && pwd)"
LOCAL_PAYLOAD_SOURCE="$PROJECT_ROOT/$PAYLOAD_FILE"
LOCAL_DOC_SOURCE="$PROJECT_ROOT/$DOC_FILE"
if [[ ! -f "$LOCAL_PAYLOAD_SOURCE" ]]; then
    echo "missing payload file: $LOCAL_PAYLOAD_SOURCE" >&2
    exit 1
fi
if [[ ! -f "$LOCAL_DOC_SOURCE" ]]; then
    echo "missing doc file: $LOCAL_DOC_SOURCE" >&2
    exit 1
fi

LOCAL_PAYLOAD="$LOCAL_DIR/rag_payload_${WORKLOAD}.json"
PHONE_DOC="$PHONE_DOC_DIR/doc_${WORKLOAD}.txt"
LOCAL_DOC_TMP="$LOCAL_TMP_DIR/doc_${WORKLOAD}.txt"

# Resolve profile -> mode + backends
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
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "try: pure_cpu_sequential | pure_npu_sequential | npu_cpu" >&2
    exit 2
    ;;
esac

echo "=== workload=$WORKLOAD profile=$PROFILE mode=$MODE prefill=$GEN_PREFILL_BACKEND decode=$GEN_DECODE_BACKEND device=$DEVICE_PROFILE serial=$SERIAL ==="

echo "[1/4] create phone doc dir"
adb -s "$SERIAL" shell "mkdir -p $PHONE_DOC_DIR"

echo "[2/4] push doc to phone"
# Read doc from workloads dir and push to phone
mkdir -p "$LOCAL_TMP_DIR"
python3 - <<PY
from pathlib import Path
src = Path('$LOCAL_DOC_SOURCE')
doc = src.read_text(encoding='utf-8')
phone_doc = Path('$LOCAL_DOC_TMP')
phone_doc.write_text(doc, encoding='utf-8')
print(f'doc extracted: {phone_doc} ({len(doc)} chars)')
PY
adb -s "$SERIAL" push "$LOCAL_DOC_TMP" "$PHONE_DOC" >/dev/null

echo "[3/4] build payload"
LOCAL_PAYLOAD_SOURCE_FOR_PY="$LOCAL_PAYLOAD_SOURCE" \
LOCAL_PAYLOAD_FOR_PY="$LOCAL_PAYLOAD" \
LOCAL_DOC_SOURCE_FOR_PY="$LOCAL_DOC_SOURCE" \
RAG_MODE="$MODE" \
RAG_PREFILL_BACKEND="$GEN_PREFILL_BACKEND" \
RAG_DECODE_BACKEND="$GEN_DECODE_BACKEND" \
python3 - <<'PY'
import json
import os
from pathlib import Path

src_payload = Path(os.environ['LOCAL_PAYLOAD_SOURCE_FOR_PY'])
local_payload = Path(os.environ['LOCAL_PAYLOAD_FOR_PY'])
doc_source = Path(os.environ['LOCAL_DOC_SOURCE_FOR_PY'])
rag_mode = os.environ['RAG_MODE']
rag_prefill_backend = os.environ['RAG_PREFILL_BACKEND']
rag_decode_backend = os.environ['RAG_DECODE_BACKEND']

payload = json.loads(src_payload.read_text(encoding='utf-8'))
payload['doc'] = doc_source.read_text(encoding='utf-8')
payload['mode'] = rag_mode
payload['generation_prefill_backend'] = rag_prefill_backend
payload['generation_decode_backend'] = rag_decode_backend
# Ensure decode steps and max_tokens are consistent
payload['generation_decode_steps'] = 64
payload['max_tokens'] = 64

local_payload.write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding='utf-8',
)
print(f'payload ready: {local_payload}')
PY

echo "[4/4] run request"
adb -s "$SERIAL" forward tcp:18080 tcp:8080 >/dev/null
curl -sS -X POST http://127.0.0.1:18080/v1/rag \
  -H "Content-Type: application/json" \
  --data-binary @"$LOCAL_PAYLOAD"

echo
echo "done"
echo "phone doc path: $PHONE_DOC"
echo "local payload:  $LOCAL_PAYLOAD"
echo "device flag: $DEVICE_PROFILE"
echo "serial: $SERIAL"
