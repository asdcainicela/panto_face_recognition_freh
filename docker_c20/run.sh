#!/bin/bash
set -e

CONTAINER_NAME="l4t-cpp-20"
IMAGE_NAME="l4t-cpp20"
WORKSPACE_DIR="${HOME}/jetson_workspace"

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

echo "=== Jetson Container Setup ==="

[ ! -d "$WORKSPACE_DIR" ] && mkdir -p "$WORKSPACE_DIR"

export DISPLAY=${DISPLAY:-:0}
XAUTH="${HOME}/.docker.xauth"

rm -f "$XAUTH" 2>/dev/null || true
touch "$XAUTH" 2>/dev/null && chmod 600 "$XAUTH"
xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        docker start ${CONTAINER_NAME} >/dev/null
        sleep 2
    fi
else
    docker run -d \
      --runtime nvidia \
      --gpus all \
      --name ${CONTAINER_NAME} \
      --restart unless-stopped \
      --network host \
      --ipc=host \
      --pid=host \
      -e USER_ID=$USER_ID \
      -e GROUP_ID=$GROUP_ID \
      -e USER_NAME=$USER_NAME \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTH \
      -e QT_X11_NO_MITSHM=1 \
      -e XDG_RUNTIME_DIR=/tmp \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:$LD_LIBRARY_PATH \
      -e GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0 \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v $XAUTH:$XAUTH:rw \
      -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:rw \
      -v /usr/lib/aarch64-linux-gnu/tegra-egl:/usr/lib/aarch64-linux-gnu/tegra-egl:rw \
      -v "${WORKSPACE_DIR}":/workspace:rw \
      -v /dev:/dev:rw \
      -p 8888:8888 \
      --privileged \
      --device=/dev/video0 \
      --device=/dev/video1 \
      --device=/dev/nvhost-ctrl \
      --device=/dev/nvhost-ctrl-gpu \
      --device=/dev/nvhost-prof-gpu \
      --device=/dev/nvmap \
      --device=/dev/nvhost-gpu \
      --device=/dev/nvhost-as-gpu \
      --device=/dev/nvhost-vic \
      ${IMAGE_NAME} \
      /bin/bash -c "sleep infinity"

    docker exec ${CONTAINER_NAME} bash -c "
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq \
            libgtk2.0-dev libgtk-3-dev libglib2.0-0 libsm6 libxext6 \
            libxrender1 libgomp1 libgl1-mesa-glx libgles2-mesa libegl1-mesa \
            sudo libspdlog-dev git nano vim cmake build-essential wget curl htop tree x11-apps
        ldconfig
        git config --global user.email 'asdcainicela@gmail.com'
        git config --global user.name 'asdcainicela'
        git config --global --add safe.directory '*'
        chown -R $USER_ID:$GROUP_ID /workspace
        cat > /etc/profile.d/jetson-env.sh << 'EOF'
export DISPLAY=\${DISPLAY:-:0}
export XAUTHORITY=\${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:\$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
EOF
        chmod +x /etc/profile.d/jetson-env.sh
    " >/dev/null 2>&1

    docker exec ${CONTAINER_NAME} bash -c "
        cd /workspace
        git clone https://asdcainicela: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@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || true
        git clone https://asdcainicela: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@github.com/asdcainicela/panto_face_recognition_freh.git 2>/dev/null || true
        chown -R $USER_ID:$GROUP_ID /workspace
    " >/dev/null 2>&1
fi

# Iniciar Jupyter Lab si no está corriendo
if ! docker exec ${CONTAINER_NAME} pgrep -f "jupyter-lab" >/dev/null 2>&1; then
    echo "Iniciando Jupyter Lab..."
    docker exec -d ${CONTAINER_NAME} bash -lc "mkdir -p /var/log && jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser >> /var/log/jupyter.log 2>&1 &"
    sleep 3
else
    echo "Jupyter Lab ya está corriendo"
fi

echo ""
echo "Contenedor Jetson listo"
echo ""
echo "Servicios activos:"
echo "  Jupyter Lab: http://localhost:8888"
echo "  Password: nvidia"
echo ""
echo "Comandos útiles:"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo "  docker logs -f ${CONTAINER_NAME}"
echo "  docker exec -it ${CONTAINER_NAME} tail -f /var/log/jupyter.log"
