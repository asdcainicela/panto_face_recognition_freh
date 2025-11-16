#!/bin/bash
# Convertir ONNX a TensorRT Engine usando trtexec (herramienta NVIDIA)

set -e

MODELS_DIR="models"
ONNX_FILE="${MODELS_DIR}/retinaface.onnx"
ENGINE_FILE="${MODELS_DIR}/retinaface_fp16.engine"
WORKSPACE_MB=3072  # 3GB para 8GB RAM

echo "======================================"
echo "  ONNX → TensorRT Conversion"
echo "======================================"
echo ""

if [ ! -f "$ONNX_FILE" ]; then
    echo "❌ No se encontró: $ONNX_FILE"
    echo "   Descarga los modelos primero"
    exit 1
fi

ONNX_SIZE=$(du -h "$ONNX_FILE" | cut -f1)
echo "📦 Modelo ONNX: $ONNX_SIZE"
echo "💾 Workspace: ${WORKSPACE_MB}MB"
echo ""

if ! command -v trtexec &> /dev/null; then
    echo "❌ trtexec no encontrado"
    echo ""
    echo "Instalación en Jetson:"
    echo "  sudo apt install tensorrt"
    echo "  # trtexec está en /usr/src/tensorrt/bin/trtexec"
    exit 1
fi

echo "🔧 Construyendo engine (esto toma 5-15 min)..."
echo ""

# Usar trtexec para conversión
trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 \
    --workspace=${WORKSPACE_MB} \
    --minShapes=input:1x3x320x320 \
    --optShapes=input:1x3x640x640 \
    --maxShapes=input:1x3x1280x1280 \
    --verbose

if [ $? -eq 0 ]; then
    ENGINE_SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
    echo ""
    echo "======================================"
    echo "  ✅ CONVERSIÓN EXITOSA"
    echo "======================================"
    echo "Engine: $ENGINE_SIZE"
    echo "Ubicación: $ENGINE_FILE"
    echo ""
    echo "Performance esperado:"
    echo "  - ONNX CUDA: ~40-60ms"
    echo "  - TensorRT FP16: ~8-15ms ⚡⚡⚡"
    echo ""
else
    echo ""
    echo "❌ Error en conversión"
    exit 1
fi