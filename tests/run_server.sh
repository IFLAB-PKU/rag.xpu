#!/usr/bin/env bash

SERIAL="3B15940035V00000"
PORT=8080
SERVER_BIN="powerserve-server"


adb -s $SERIAL shell -T "
    export LD_LIBRARY_PATH=/data/local/tmp/shuhua/models/lib:/data/local/tmp/shuhua/models/qnn_libs
    cd /data/local/tmp/shuhua/
    ./models/bin/$SERVER_BIN -d ./models --port $PORT
"

# Note: Known server output truncation issue, inconvenient for profiling.
