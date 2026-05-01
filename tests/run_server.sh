#!/usr/bin/env bash

SERIAL="3B15940035V00000"
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

echo "server_profile=$SERVER_PROFILE server_bin=$SERVER_BIN"


adb -s $SERIAL shell -T "
    export LD_LIBRARY_PATH=/data/local/tmp/shuhua/models/lib:/data/local/tmp/shuhua/models/qnn_libs
    cd /data/local/tmp/shuhua/
    ./models/bin/$SERVER_BIN -d ./models --port $PORT
"

# How to run
# ./tests/run_server.sh cpu
# ./tests/run_server.sh npu

# Note: Known server output truncation issue, inconvenient for profiling.
