#!/usr/bin/env bash
set -euo pipefail

SERIAL="3B15940035V00000"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DOC="$LOCAL_DIR/rag_long_doc.txt"
LOCAL_PAYLOAD="$LOCAL_DIR/rag_payload_once.json"
PHONE_DOC_DIR="/data/local/tmp/shuhua/doc"
PHONE_DOC="$PHONE_DOC_DIR/rag_long_doc.txt"

if [[ ! -f "$LOCAL_DOC" ]]; then
  echo "missing doc: $LOCAL_DOC" >&2
  exit 1
fi

echo "[1/4] create phone doc dir"
adb -s "$SERIAL" shell "mkdir -p $PHONE_DOC_DIR"

echo "[2/4] push long doc to phone"
adb -s "$SERIAL" push "$LOCAL_DOC" "$PHONE_DOC" >/dev/null

echo "[3/4] build one-shot payload"
LOCAL_DOC_FOR_PY="$LOCAL_DOC" LOCAL_PAYLOAD_FOR_PY="$LOCAL_PAYLOAD" python3 - <<'PY'
import json
import os
from pathlib import Path

local_doc = os.environ['LOCAL_DOC_FOR_PY']
local_payload = os.environ['LOCAL_PAYLOAD_FOR_PY']

payload = {
    'doc': Path(local_doc).read_text(encoding='utf-8', errors='ignore'),
    'query': 'OpenAI的发展中体现了哪些取舍？',
    'mode': 'sequential',
    'generation_model': 'qwen3-0.6b-base',
    'embedding_model': 'qwen3-embedding-0.6b',
    'rerank_model': 'qwen3-reranker-0.6b',
    'enable_query_expansion': True,
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
