#!/bin/bash
set -e

CONTAINER_NAME="${CONTAINER_NAME:-l4t-cpp-20}"
IMAGE_NAME="${IMAGE_NAME:-l4t-cpp20}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${HOME}/jetson_workspace}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-nvidia}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-gerald.cainicela.a@gmail.com}"
GIT_USER_NAME="${GIT_USER_NAME:-user-asd}"
CUDA_VERSION="${CUDA_VERSION:-11.4}"

# =============================================================================
# SETUP
# =============================================================================
[ ! -d "$WORKSPACE_DIR" ] && mkdir -p "$WORKSPACE_DIR"

export DISPLAY=${DISPLAY:-:0}
XAUTH="${HOME}/.docker.xauth"

rm -f "$XAUTH" 2>/dev/null || true
touch "$XAUTH" && chmod 600 "$XAUTH"
xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

ENTRYPOINT_SCRIPT="${WORKSPACE_DIR}/container_startup.sh"
cat > "$ENTRYPOINT_SCRIPT" << EOF
#!/bin/bash
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\${LD_LIBRARY_PATH}
export DISPLAY=\${DISPLAY:-:0}
export XAUTHORITY=\${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:\$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
mkdir -p /var/log
nohup jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT} --allow-root --no-browser --NotebookApp.token='${JUPYTER_TOKEN}' > /var/log/jupyter.log 2>&1 &
sleep infinity
EOF
chmod +x "$ENTRYPOINT_SCRIPT"

# Crear contenedor si no existe
if [ ! "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    docker run -d \
      --runtime nvidia \
      --gpus all \
      --name ${CONTAINER_NAME} \
      --restart unless-stopped \
      --network host \
      --ipc=host \
      --pid=host \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTH \
      -e QT_X11_NO_MITSHM=1 \
      -e XDG_RUNTIME_DIR=/tmp \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v $XAUTH:$XAUTH:rw \
      -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:rw \
      -v /usr/lib/aarch64-linux-gnu/tegra-egl:/usr/lib/aarch64-linux-gnu/tegra-egl:rw \
      -v "${WORKSPACE_DIR}":/workspace:rw \
      -v /dev:/dev:rw \
      --privileged \
      ${IMAGE_NAME} \
      /workspace/container_startup.sh >/dev/null 2>&1

    docker exec -u root ${CONTAINER_NAME} bash -c "
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq sudo git nano vim cmake build-essential wget curl htop tree \
            libgtk2.0-dev libgtk-3-dev libglib2.0-0 libsm6 libxext6 libxrender1 \
            libgomp1 libgl1-mesa-glx libgles2-mesa libegl1-mesa python3-pip x11-apps
        ldconfig
        pip3 install --quiet jupyterlab 2>/dev/null || true
        git config --global user.email '${GIT_USER_EMAIL}'
        git config --global user.name '${GIT_USER_NAME}'
        git config --global --add safe.directory '*'
        cat > /etc/profile.d/cuda-env.sh << 'CUDAEOF'
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\\\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\\\$LD_LIBRARY_PATH
CUDAEOF
        chmod +x /etc/profile.d/cuda-env.sh
        cat >> /root/.bashrc << 'BASHEOF'
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\\\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\\\$LD_LIBRARY_PATH
BASHEOF
    " >/dev/null 2>&1
fi

# Iniciar si está detenido
[ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ] && docker start ${CONTAINER_NAME} >/dev/null 2>&1

# Entrar directo
exec docker exec -it -u root -w /workspace ${CONTAINER_NAME} bash