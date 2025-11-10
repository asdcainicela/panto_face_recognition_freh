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

# Configurar X11
rm -f "$XAUTH" 2>/dev/null || true
touch "$XAUTH" && chmod 600 "$XAUTH"
xauth nlist $DISPLAY 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

# Crear o levantar contenedor
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "Iniciando contenedor existente..."
        docker start ${CONTAINER_NAME} >/dev/null
        sleep 2
    fi
else
    echo "Creando contenedor..."
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
      /bin/bash -c "sleep infinity"
fi

# Setup completo del contenedor
echo "Configurando contenedor..."
docker exec -u root ${CONTAINER_NAME} bash -c "
    export DEBIAN_FRONTEND=noninteractive
    
    # Instalar paquetes básicos
    apt-get update -qq
    apt-get install -y -qq sudo git nano vim cmake build-essential wget curl htop tree \
        libgtk2.0-dev libgtk-3-dev libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgomp1 libgl1-mesa-glx libgles2-mesa libegl1-mesa python3-pip x11-apps
    ldconfig
    
    # Jupyter
    pip3 install --quiet jupyterlab 2>/dev/null || true
    
    # Crear usuario con el mismo UID/GID del host
    if ! getent group $GROUP_ID >/dev/null 2>&1; then
        groupadd -g $GROUP_ID $USER_NAME 2>/dev/null || true
    fi
    
    if ! id -u $USER_ID >/dev/null 2>&1; then
        useradd -u $USER_ID -g $GROUP_ID -G video,sudo -m -s /bin/bash $USER_NAME
        mkdir -p /home/$USER_NAME
        chown -R $USER_ID:$GROUP_ID /home/$USER_NAME
    fi
    
    # Configurar sudo sin password
    echo '$USER_NAME ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/$USER_NAME
    chmod 440 /etc/sudoers.d/$USER_NAME
    
    # Permisos en dispositivos
    chmod 666 /dev/video* 2>/dev/null || true
    chmod 666 /dev/nvhost-* 2>/dev/null || true
    chmod 666 /dev/nvmap 2>/dev/null || true
    
    # Git config
    sudo -u $USER_NAME git config --global user.email 'userasd@gmail.com'
    sudo -u $USER_NAME git config --global user.name 'userasd'
    sudo -u $USER_NAME git config --global --add safe.directory '*'
    
    # Workspace
    mkdir -p /workspace
    chown -R $USER_ID:$GROUP_ID /workspace
    
    # Variables de entorno
    cat > /etc/profile.d/jetson-env.sh << 'ENVEOF'
export DISPLAY=\${DISPLAY:-:0}
export XAUTHORITY=\${XAUTHORITY:-/tmp/.docker.xauth}
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl:\$LD_LIBRARY_PATH
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
ENVEOF
    chmod +x /etc/profile.d/jetson-env.sh
    
    echo 'Setup completado'
"

# Jupyter en background
echo "Iniciando Jupyter Lab..."
docker exec -u root -w /workspace ${CONTAINER_NAME} bash -c "
    mkdir -p /var/log
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='nvidia' > /var/log/jupyter.log 2>&1 &
" 2>/dev/null || true

echo ""
echo "✓ Contenedor listo"
echo ""
echo "Servicios:"
echo "  → Jupyter Lab: http://localhost:8888/?token=nvidia"
echo ""
echo "Entrando al contenedor..."
docker exec -it -u $USER_NAME -w /workspace ${CONTAINER_NAME} bash