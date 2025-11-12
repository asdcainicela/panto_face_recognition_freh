#!/bin/bash
set -e

echo "[INFO] Descargando modelos ONNX..."

# 1. RetinaFace (detección rostros) - ~27 MB
if [ ! -f "retinaface.onnx" ]; then
    echo "  - Descargando RetinaFace..."
    wget -q --show-progress \
        https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx \
        -O retinaface.onnx
else
    echo "  - RetinaFace ya existe"
fi

# 2. ArcFace R100 (reconocimiento) - ~250 MB
if [ ! -f "arcface_r100.onnx" ]; then
    echo "  - Descargando ArcFace R100..."
    wget -q --show-progress \
        https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx \
        -O arcface_r100.onnx
else
    echo "  - ArcFace R100 ya existe"
fi

# 3. Real-ESRGAN x4 (super-resolución) - ~67 MB
if [ ! -f "realesr_x4.onnx" ]; then
    echo "  - Descargando Real-ESRGAN x4..."
    wget -q --show-progress \
        https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx \
        -O realesr_x4.onnx
else
    echo "  - Real-ESRGAN x4 ya existe"
fi

echo ""
echo "[INFO] Modelos descargados:"
ls -lh *.onnx 2>/dev/null || echo "  - Ninguno encontrado"

echo ""
echo "[INFO] Verificando TensorRT..."
if command -v trtexec &>/dev/null; then
    echo "  - TensorRT encontrado. Puedes convertir con:"
    echo "    trtexec --onnx=modelo.onnx --saveEngine=modelo.trt --fp16"
else
    echo "  - TensorRT no encontrado. Usando ONNX Runtime."
fi

echo ""
echo "[DONE] Configuración completada."
echo "       Ejecuta desde raíz: ./build.sh"