#!/usr/bin/env bash
set -euo pipefail

# Benchmark all workloads (4K/6K/8K/10K) with a given profile.
# Usage:
#   ./tests/run_rag_benchmark.sh [profile]
#
# Examples:
#   ./tests/run_rag_benchmark.sh npu_cpu
#   ./tests/run_rag_benchmark.sh npu_cpu_cs
#   ./tests/run_rag_benchmark.sh pure_cpu_sequential
#   ./tests/run_rag_benchmark.sh pure_npu_sequential

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${1:-npu_cpu}"

WORKLOADS=("4K" "6K" "8K" "10K")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$LOCAL_DIR/results"
RESULT_FILE="$RESULT_DIR/benchmark_${PROFILE}_${TIMESTAMP}.json"

mkdir -p "$RESULT_DIR"

echo "========================================"
echo "RAG Workload Benchmark"
echo "profile: $PROFILE"
echo "timestamp: $TIMESTAMP"
echo "========================================"

# Array to hold per-workload results
results_json="["
first=true

for workload in "${WORKLOADS[@]}"; do
    echo ""
    echo ">>> Running workload: $workload"

    # Run the workload and capture raw JSON response
    response=$("$LOCAL_DIR/run_rag_workload.sh" "$workload" "$PROFILE")

    # Extract stage metrics using python
    metrics=$(python3 -c "
import json, sys
# Find the JSON object in the response (last line that looks like JSON)
lines = sys.stdin.read().split('\n')
data = None
for line in reversed(lines):
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        try:
            data = json.loads(line)
            break
        except:
            continue
if data is None:
    print('{}')
    sys.exit(0)

metrics = data.get('stage_metrics_ms', {})
out = {
    'workload': '$workload',
    'mode_requested': data.get('mode_requested', ''),
    'mode_used': data.get('mode_used', ''),
    'generation_prefill_backend_target': data.get('debug', {}).get('generation_prefill_backend_target', ''),
    'generation_decode_backend_target': data.get('debug', {}).get('generation_decode_backend_target', ''),
    'generation_segmented_prefill_used': data.get('debug', {}).get('generation_segmented_prefill_used', False),
    'indexing_ms': metrics.get('indexing', 0),
    'query_embedding_ms': metrics.get('query_embedding', 0),
    'embedding_ms': metrics.get('embedding', 0),
    'query_expand_ms': metrics.get('query_expand', 0),
    'searching_ms': metrics.get('searching', 0),
    'reranking_ms': metrics.get('reranking', 0),
    'generation_ms': metrics.get('generation', 0),
    'total_ms': metrics.get('total', 0),
    'answer': data.get('answer', '')[:200],
}
print(json.dumps(out, ensure_ascii=False))
" <<< "$response")

    if [[ "$first" == "true" ]]; then
        first=false
    else
        results_json+=","
    fi
    results_json+="$metrics"

done

results_json+="]"

# Write result file
echo "$results_json" > "$RESULT_FILE"

echo ""
echo "========================================"
echo "Benchmark complete"
echo "Result file: $RESULT_FILE"
echo "========================================"

# Pretty print summary
python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
print('{:<8} {:<18} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
    'Workload', 'ModeReq/Used', 'Total(ms)', 'Indexing', 'Embedding', 'QueryEmb', 'Search', 'Rerank', 'Gen'
))
print('-' * 100)
for row in data:
    mode_pair = f\"{row['mode_requested']}/{row['mode_used']}\"
    print(f\"{row['workload']:<8} {mode_pair:<18} {row['total_ms']:<12.1f} {row['indexing_ms']:<10.1f} {row['embedding_ms']:<10.1f} {row['query_embedding_ms']:<10.1f} {row['searching_ms']:<10.1f} {row['reranking_ms']:<10.1f} {row['generation_ms']:<10.1f}\")
"
