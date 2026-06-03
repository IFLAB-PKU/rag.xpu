#!/usr/bin/env bash

set -euo pipefail

DEVICE_PROFILE_RAW=""
if [[ "$#" -gt 0 ]]; then
    DEVICE_PROFILE_RAW="${!#}"
fi
DEVICE_PROFILE="8gen5"
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

PORT=8080

SERVER_PROFILE="${1:-cpu}"
case "$SERVER_PROFILE" in
    cpu)
        SERVER_BIN="powerserve-server-cpu"
        ;;
    npu)
        SERVER_BIN="powerserve-server"
        ;;
    *)
        # Allow passing explicit binary names directly.
        SERVER_BIN="$SERVER_PROFILE"
        ;;
esac

echo "server_profile=$SERVER_PROFILE server_bin=$SERVER_BIN device=$DEVICE_PROFILE serial=$SERIAL"


adb -s $SERIAL shell -T "
    export LD_LIBRARY_PATH=/data/local/tmp/shuhua/models/lib:/data/local/tmp/shuhua/models/qnn_libs
    cd /data/local/tmp/shuhua/
    ./models/bin/$SERVER_BIN -d ./models --port $PORT
"

# How to run
# ./tests/run_server.sh cpu 8gen5
# ./tests/run_server.sh npu 8gen4

# Note: Known server output truncation issue, inconvenient for profiling.
