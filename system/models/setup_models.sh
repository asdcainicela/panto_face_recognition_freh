#!/bin/bash
set -e

echo "[INFO] Descargando modelos..."

download_model() {
    [ -f "$2" ] || wget -q --show-progress "$1" -O "$2"
}

download_model https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx retinaface.onnx
download_model https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx arcface_r100.onnx
download_model https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx realesr_x4.onnx

echo "[INFO] Modelos:"
ls -lh *.onnx 2>/dev/null || echo "Ninguno"

if command -v trtexec &>/dev/null; then
    echo "[INFO] TensorRT disponible. Ejemplo:"
    echo "trtexec --onnx=model.onnx --saveEngine=model.trt --fp16"
else
    echo "[INFO] TensorRT no encontrado."
fi

echo "[OK] Listo."
