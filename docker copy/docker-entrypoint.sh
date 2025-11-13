#!/bin/bash
set -e

export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

ldconfig 2>/dev/null || true

if command -v jupyter >/dev/null 2>&1; then
    # Verificar si JupyterLab ya está corriendo
    if ! pgrep -f "jupyter-lab" > /dev/null; then
        nohup jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --allow-root \
            --no-browser \
            > /var/log/jupyter.log 2>&1 &
        
        # Esperar un momento para que inicie
        sleep 2
        echo "JupyterLab start http://0.0.0.0:8888"
    else
        echo "JupyterLab running"
    fi
else
    echo "JupyterLab no found"
fi

exec "$@"