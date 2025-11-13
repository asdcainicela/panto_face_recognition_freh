#!/bin/bash
set -e

export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/usr/local/cuda/lib64:/opt/onnxruntime/lib:${LD_LIBRARY_PATH}
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

ldconfig 2>/dev/null || true

if command -v jupyter >/dev/null 2>&1; then
    nohup jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --allow-root \
        --no-browser \
        > /var/log/jupyter.log 2>&1 &
fi

exec "$@"