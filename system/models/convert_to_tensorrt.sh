#!/bin/bash
set -e

declare -A MODELOS=(
    ["retinaface.onnx"]="retinaface_fp16.engine"
    ["arcface_r100.onnx"]="arcface_r100_fp16.engine"
)

WORKSPACE_MB=3072

# Buscar trtexec
if command -v trtexec &> /dev/null; then
    TRTEXEC=$(command -v trtexec)
elif [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
    echo "trtexec no encontrado. Instala TensorRT."
    exit 1
fi

for ONNX_FILE in "${!MODELOS[@]}"; do
    ENGINE_FILE="${MODELOS[$ONNX_FILE]}"

    [ ! -f "$ONNX_FILE" ] && echo "No se encontró $ONNX_FILE" && continue
    [ -f "$ENGINE_FILE" ] && echo "Engine ya existe: $ENGINE_FILE" && continue

    echo "Convirtiendo $ONNX_FILE → $ENGINE_FILE..."
    $TRTEXEC \
        --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --workspace=${WORKSPACE_MB} \
        --minShapes=input:1x3x320x320 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x1280x1280 > /dev/null

    echo "Hecho: $ENGINE_FILE"
done
