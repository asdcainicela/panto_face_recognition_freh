#!/bin/bash
CONTAINER_NAME="l4t-cpp-20"

if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Contenedor no está corriendo"
    exit 1
fi

docker exec -it -u root -w /workspace ${CONTAINER_NAME} bash