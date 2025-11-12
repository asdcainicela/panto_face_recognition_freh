#!/bin/bash
set -e

echo "[INFO] Descargando modelos ONNX..."

# 1. RetinaFace (detección rostros) - ~105 MB
if [ ! -f "retinaface.onnx" ] || [ ! -s "retinaface.onnx" ]; then
    echo "  - Descargando RetinaFace..."
    wget -q --show-progress \
        https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx \
        -O retinaface.onnx
else
    echo "  - RetinaFace OK ($(du -h retinaface.onnx | cut -f1))"
fi

# 2. ArcFace R100 (reconocimiento) - ~131 MB
if [ ! -f "arcface_r100.onnx" ] || [ ! -s "arcface_r100.onnx" ]; then
    echo "  - Descargando ArcFace R100..."
    wget -q --show-progress \
        https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx \
        -O arcface_r100.onnx
else
    echo "  - ArcFace R100 OK ($(du -h arcface_r100.onnx | cut -f1))"
fi

# 3. Real-ESRGAN x4 (super-resolución) - ~67 MB
# URL alternativa verificada
if [ ! -f "realesr_x4.onnx" ] || [ ! -s "realesr_x4.onnx" ]; then
    echo "  - Descargando Real-ESRGAN x4..."
    
    # Probar varias fuentes
    if ! wget -q --show-progress \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx \
        -O realesr_x4.onnx 2>/dev/null; then
        
        echo "  - Fuente 1 falló, probando fuente 2..."
        wget -q --show-progress \
            https://huggingface.co/rockeycoss/RealESRGAN/resolve/main/RealESRGAN_x4plus.onnx \
            -O realesr_x4.onnx
    fi
else
    echo "  - Real-ESRGAN x4 OK ($(du -h realesr_x4.onnx | cut -f1))"
fi

echo ""
echo "[INFO] Verificando modelos descargados:"
echo "========================================"

TOTAL_SIZE=0
ERROR_COUNT=0

for model in retinaface.onnx arcface_r100.onnx realesr_x4.onnx; do
    if [ -f "$model" ] && [ -s "$model" ]; then
        SIZE=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)
        SIZE_MB=$((SIZE / 1024 / 1024))
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE_MB))
        echo "  [OK] $model - ${SIZE_MB}MB"
    else
        echo "  [FAIL] $model - FALTA O VACIO"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
done

echo "========================================"
echo "Total descargado: ${TOTAL_SIZE}MB"

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "[ERROR] $ERROR_COUNT modelo(s) fallaron."
    echo "Ejecuta nuevamente o descarga manual:"
    echo "  RetinaFace: https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx"
    echo "  ArcFace: https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"
    echo "  RealESRGAN: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx"
    exit 1
fi

echo ""
echo "[INFO] TensorRT (opcional):"
if command -v trtexec &>/dev/null; then
    echo "  - TensorRT encontrado. Para convertir:"
    echo "    trtexec --onnx=modelo.onnx --saveEngine=modelo.trt --fp16 --workspace=2048"
else
    echo "  - TensorRT no disponible (usará ONNX Runtime)"
fi

echo ""
echo "[DONE] Modelos listos para usar."
echo "       Ejecuta desde raíz: ./build.sh && ./run.sh 8"