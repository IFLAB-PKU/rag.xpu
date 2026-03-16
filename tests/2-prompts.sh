#!/system/bin/sh

URL="${URL:-http://localhost:8080/completion}"
MODEL="${MODEL:-qwen3-0.6b-base}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CURL_MAX_TIME="${CURL_MAX_TIME:-180}"
OUT_DIR="${OUT_DIR:-/data/local/tmp/shuhua}"
MODE="${MODE:-both}"
ROUNDS="${ROUNDS:-1}"
SERIAL_DELAY_SEC="${SERIAL_DELAY_SEC:-5}"
OVERLAP_DELAY_SEC="${OVERLAP_DELAY_SEC:-0.2}"

PROMPT1="Write a long, immersive short story about a weary but determined traveler named Kael, who must cross the vast, unforgiving Sunstone Desert to reach the mythical oasis of Elara before a pursuing sandstorm catches him. The story should be at least 800 words long. Include vivid sensory details about the scorching heat, the shifting dunes, the silence of the night, and the mirages he encounters. Develop his internal monologue as he struggles with thirst, exhaustion, and doubt, but is driven by a mysterious, old map he found. Introduce a brief, meaningful dialogue with a ghostly figure he meets at an ancient, half-buried ruin, who offers him a cryptic choice. The story should have a clear narrative arc with a tense climax as the storm closes in and a rewarding, thought-provoking conclusion at the oasis."

PROMPT2="Provide a detailed, step-by-step explanation of the Transformer model's attention mechanism, specifically scaled dot-product attention and multi-head attention, designed for absolute beginners with no prior deep learning knowledge. Start with the fundamental intuition: why attention is needed in sequence tasks (solving the long-range dependency problem of RNNs). Use a simple, concrete example sentence like 'The animal didn't cross the street because it was too tired' to illustrate how attention helps the model understand what 'it' refers to. Break down the process into clear, digestible steps: 1) Creating Queries, Keys, and Values from input embeddings. 2) Calculating attention scores via dot products. 3) Applying the softmax function to get attention weights. 4) Using the weights to get a weighted sum of the Values. 5) Explain the role of the scaling factor. 6) Introduce multi-head attention as running this process in parallel, allowing the model to focus on different types of relationships (e.g., syntactic vs. semantic). Use analogies like a search engine or a spotlight to make the concepts relatable. Include simple, illustrative calculations with hypothetical small vectors to demonstrate the matrix operations without overwhelming the reader. Conclude by briefly connecting attention to the overall Transformer architecture (encoder-decoder structure)."

RUN_TAG_BASE="$(date +%Y%m%d-%H%M%S)"
REPORT_CSV="${OUT_DIR}/pd-bench-${RUN_TAG_BASE}.csv"

REQ1=$(printf '{"prompt":"%s","max_tokens":%s,"model":"%s"}' "$PROMPT1" "$MAX_TOKENS" "$MODEL")
REQ2=$(printf '{"prompt":"%s","max_tokens":%s,"model":"%s"}' "$PROMPT2" "$MAX_TOKENS" "$MODEL")

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$1"
}

pid1=""
pid2=""

cleanup() {
  if [ -n "$pid1" ]; then
    kill "$pid1" 2>/dev/null || true
  fi
  if [ -n "$pid2" ]; then
    kill "$pid2" 2>/dev/null || true
  fi
}

trap cleanup INT TERM HUP EXIT

run_req() {
  req_name="$1"
  req_payload="$2"
  req_output="$3"
  req_time_file="$4"

  log "${req_name} start"
  curl -sS --max-time "$CURL_MAX_TIME" --request POST --url "$URL" \
    --header "Content-Type: application/json" \
    --data "$req_payload" \
    --output "$req_output" \
    --write-out "%{time_total}" > "$req_time_file"
  rc=$?

  req_time="NA"
  if [ -f "$req_time_file" ]; then
    req_time="$(cat "$req_time_file")"
  fi

  if [ "$rc" -eq 0 ]; then
    log "${req_name} done time_s=${req_time}"
  else
    log "${req_name} failed rc=${rc} time_s=${req_time}"
  fi

  return "$rc"
}

run_once() {
  mode_name="$1"
  round_id="$2"
  delay_sec="$3"

  run_tag="${RUN_TAG_BASE}-${mode_name}-r${round_id}"
  req1_out="${OUT_DIR}/req1-${run_tag}.json"
  req2_out="${OUT_DIR}/req2-${run_tag}.json"
  req1_time_file="${OUT_DIR}/req1-${run_tag}.time"
  req2_time_file="${OUT_DIR}/req2-${run_tag}.time"

  start_ts=$(date +%s)
  log "mode=${mode_name} round=${round_id} start delay=${delay_sec}s"

  run_req "req1" "$REQ1" "$req1_out" "$req1_time_file" &
  pid1=$!

  sleep "$delay_sec"

  run_req "req2" "$REQ2" "$req2_out" "$req2_time_file" &
  pid2=$!

  wait "$pid1"
  rc1=$?
  wait "$pid2"
  rc2=$?

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  req1_time="NA"
  req2_time="NA"
  [ -f "$req1_time_file" ] && req1_time="$(cat "$req1_time_file")"
  [ -f "$req2_time_file" ] && req2_time="$(cat "$req2_time_file")"

  log "mode=${mode_name} round=${round_id} done rc1=${rc1} rc2=${rc2} elapsed_s=${elapsed}"
  log "outputs: ${req1_out} ${req2_out}"
  log "timings: req1_s=${req1_time} req2_s=${req2_time}"

  echo "${mode_name},${round_id},${delay_sec},${rc1},${rc2},${elapsed},${req1_time},${req2_time},${req1_out},${req2_out}" >> "$REPORT_CSV"

  LAST_ELAPSED="$elapsed"

  if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
    return 1
  fi
  return 0
}

run_mode() {
  mode_name="$1"
  delay_sec="$2"
  i=1
  elapsed_sum=0
  ok_count=0
  while [ "$i" -le "$ROUNDS" ]; do
    run_once "$mode_name" "$i" "$delay_sec" || return 1
    elapsed_sum=$((elapsed_sum + LAST_ELAPSED))
    ok_count=$((ok_count + 1))
    i=$((i + 1))
  done

  mean_elapsed=0
  if [ "$ok_count" -gt 0 ]; then
    mean_elapsed=$((elapsed_sum / ok_count))
  fi
  log "mode=${mode_name} summary rounds=${ok_count} mean_elapsed_s=${mean_elapsed}"

  if [ "$mode_name" = "serial" ]; then
    SERIAL_MEAN_ELAPSED="$mean_elapsed"
  fi
  if [ "$mode_name" = "overlap" ]; then
    OVERLAP_MEAN_ELAPSED="$mean_elapsed"
  fi

  return 0
}

mkdir -p "$OUT_DIR"
echo "mode,round,delay_s,rc1,rc2,elapsed_s,req1_time_s,req2_time_s,req1_output,req2_output" > "$REPORT_CSV"

case "$MODE" in
  serial)
    run_mode "serial" "$SERIAL_DELAY_SEC"
    final_rc=$?
    ;;
  overlap)
    run_mode "overlap" "$OVERLAP_DELAY_SEC"
    final_rc=$?
    ;;
  both)
    run_mode "serial" "$SERIAL_DELAY_SEC"
    rc_serial=$?
    run_mode "overlap" "$OVERLAP_DELAY_SEC"
    rc_overlap=$?
    if [ "$rc_serial" -ne 0 ] || [ "$rc_overlap" -ne 0 ]; then
      final_rc=1
    else
      final_rc=0
    fi

    if [ "$rc_serial" -eq 0 ] && [ "$rc_overlap" -eq 0 ] && [ "$SERIAL_MEAN_ELAPSED" -gt 0 ]; then
      delta_pct=$(( (OVERLAP_MEAN_ELAPSED - SERIAL_MEAN_ELAPSED) * 100 / SERIAL_MEAN_ELAPSED ))
      log "comparison: overlap_vs_serial_delta_pct=${delta_pct}% (serial_mean_s=${SERIAL_MEAN_ELAPSED}, overlap_mean_s=${OVERLAP_MEAN_ELAPSED})"
    else
      log "comparison skipped: serial_or_overlap_failed (rc_serial=${rc_serial}, rc_overlap=${rc_overlap})"
    fi
    ;;
  *)
    log "invalid MODE=${MODE}, expected serial|overlap|both"
    final_rc=2
    ;;
esac

trap - INT TERM HUP EXIT
log "report_csv: ${REPORT_CSV}"
exit "$final_rc"