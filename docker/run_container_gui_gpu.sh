#!/bin/bash
set -e

CONTAINER_NAME="l4t-cpp"
IMAGE_NAME="l4t-base-cpp"
WORKSPACE_DIR="${HOME}/jetson_workspace"

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Jetson Container Setup ===${NC}"

if [ ! -d "$WORKSPACE_DIR" ]; then
    mkdir -p "$WORKSPACE_DIR"
fi

export DISPLAY=${DISPLAY:-:0}
XAUTH="${HOME}/.docker.xauth"

rm -f "$XAUTH" 2>/dev/null || true
touch "$XAUTH" 2>/dev/null && chmod 600 "$XAUTH"
xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo -e "${GREEN}Contenedor ya corriendo${NC}"
    else
        echo -e "${YELLOW}Iniciando contenedor...${NC}"
        docker start ${CONTAINER_NAME}
        sleep 2
    fi
else
    echo -e "${YELLOW}Creando contenedor...${NC}"
    
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
      /bin/bash -c "tail -f /dev/null"
    
    sleep 3
    
    echo -e "${YELLOW}Configurando contenedor...${NC}"
    docker exec ${CONTAINER_NAME} bash -c "
        export DEBIAN_FRONTEND=noninteractive
        
        apt-get update -qq
        
        apt-get install -y -qq \
            libgtk2.0-dev libgtk-3-dev libglib2.0-0 libsm6 libxext6 \
            libxrender1 libgomp1 libgl1-mesa-glx libgles2-mesa libegl1-mesa \
            sudo libspdlog-dev
        
        apt-get install -y -qq \
            git nano vim cmake build-essential wget curl htop tree x11-apps
        
        ldconfig
        
        groupadd -g $GROUP_ID $USER_NAME 2>/dev/null || true
        useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USER_NAME 2>/dev/null || true
        usermod -aG sudo $USER_NAME 2>/dev/null || true
        echo '$USER_NAME ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
        
        chown -R $USER_ID:$GROUP_ID /home/$USER_NAME
        chown -R $USER_ID:$GROUP_ID /workspace
        
        sudo -u $USER_NAME git config --global user.email 'asdcainicela@gmail.com'
        sudo -u $USER_NAME git config --global user.name 'asdcainicela'
        
        cat > /etc/profile.d/jetson-env.sh << 'ENVEOF'
export DISPLAY=\${DISPLAY:-:0}
export XAUTHORITY=\${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:\$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
ENVEOF
        chmod +x /etc/profile.d/jetson-env.sh
    " 2>&1 | grep -v "^Get:\|^Fetched\|^Reading" || true
    
    echo -e "${YELLOW}Clonando repositorios...${NC}"
    docker exec -u $USER_ID:$GROUP_ID ${CONTAINER_NAME} bash -c "
        cd /workspace
        git clone https://asdcainicela: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@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || echo 'lab-c-cpp ya existe'
        git clone https://asdcainicela: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@github.com/asdcainicela/panto_face_recognition_freh.git 2>/dev/null || echo 'panto_face_recognition_freh ya existe'
    "
fi

echo ""
echo -e "${GREEN}Contenedor listo${NC}"
echo -e "${BLUE}Workspace:${NC} $WORKSPACE_DIR -> /workspace"
echo -e "${BLUE}Usuario:${NC} $USER_NAME (sin problemas de permisos)"
echo ""
echo -e "${BLUE}Entrar:${NC} ${YELLOW}docker exec -it -u $USER_NAME ${CONTAINER_NAME} bash${NC}"
echo ""

read -p "¿Entrar al contenedor ahora? [s/N]: " enter_now

if [[ "$enter_now" =~ ^[Ss]$ ]]; then
    docker exec -it -u $USER_NAME ${CONTAINER_NAME} bash -c "
        source /etc/profile.d/jetson-env.sh 2>/dev/null || true
        cd /workspace
        echo 'Jetson Container - /workspace'
        exec bash
    "
else
    echo -e "${GREEN}Contenedor corriendo en background${NC}"
fi