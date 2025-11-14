#!/bin/bash
set -e

CONTAINER_NAME="l4tmlcpppy"
IMAGE_NAME="l4t-ml-cpp-py"
WORKSPACE_DIR="${HOME}/jetson_workspace"

[ ! -d "$WORKSPACE_DIR" ] && mkdir -p "$WORKSPACE_DIR"

export DISPLAY=${DISPLAY:-:0}
XAUTH="${HOME}/.docker.xauth"

rm -f "$XAUTH" 2>/dev/null || true
touch "$XAUTH" && chmod 600 "$XAUTH"
xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

if [ ! "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    docker run -d \
      --runtime nvidia \
      --name ${CONTAINER_NAME} \
      --restart unless-stopped \
      --network host \
      --ipc=host \
      --pid=host \
      -e DISPLAY=$DISPLAY \
      -e XAUTHORITY=$XAUTH \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v $XAUTH:$XAUTH:rw \
      -v "${WORKSPACE_DIR}":/workspace:rw \
      -v /dev:/dev:rw \
      --device /dev/video0:/dev/video0 \
      --device /dev/video1:/dev/video1 \
      --privileged \
      ${IMAGE_NAME}
fi

if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    docker start ${CONTAINER_NAME}
fi

exec docker exec -it -u root -w /workspace ${CONTAINER_NAME} bash