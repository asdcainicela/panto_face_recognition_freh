#!/bin/bash
# Script rápido para entrar al contenedor

CONTAINER_NAME="l4t-cpp"

# Verificar que está corriendo
if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "? Contenedor no está corriendo"
    echo "Inícialo con: ./run_container_gui_gpu.sh"
    exit 1
fi

# Configurar variables
export DISPLAY=${DISPLAY:-:0}
XAUTH=/tmp/.docker.xauth

echo "?? Entrando a ${CONTAINER_NAME}..."

# Entrar directamente
docker exec -it \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTH \
    ${CONTAINER_NAME} \
    bash -c "
    export DISPLAY=$DISPLAY
    export XAUTHORITY=$XAUTH
    cd /workspace
    
    clear
    echo '------------------------------------------------'
    echo '  ?? Jetson Container - Ready to Work'
    echo '------------------------------------------------'
    echo ''
    echo '?? Workspace: /workspace (persistente)'
    echo '? GUI + GPU habilitados'
    echo ''
    echo 'Ejemplo workflow:'
    echo '  git clone https://github.com/user/repo.git'
    echo '  cd repo && mkdir build && cd build'
    echo '  cmake .. && make && ./app'
    echo ''
    
    exec bash
"

