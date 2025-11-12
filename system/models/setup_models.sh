#!/bin/bash
set -e

echo "[INFO] Descargando modelos ONNX..."

download_if_missing() {
    local url="$1"
    local out="$2"
    local alt="$3"

    if [ ! -f "$out" ] || [ ! -s "$out" ]; then
        echo "  - Descargando $out..."
        if ! wget -q --show-progress "$url" -O "$out"; then
            if [ -n "$alt" ]; then
                echo "  - Fuente 1 falló, probando alternativa..."
                wget -q --show-progress "$alt" -O "$out" || {
                    echo "  - ERROR: No se pudo descargar $out"
                    return 1
                }
            else
                echo "  - ERROR: No se pudo descargar $out"
                return 1
            fi
        fi
    else
        echo "  - $out OK ($(du -h "$out" | cut -f1))"
    fi
}

download_if_missing \
  "https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx" \
  "retinaface.onnx"

download_if_missing \
  "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx" \
  "arcface_r100.onnx"

download_if_missing \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx" \
  "realesr_x4.onnx" \
  "https://huggingface.co/rockeycoss/RealESRGAN/resolve/main/RealESRGAN_x4plus.onnx"

echo
echo "[INFO] Verificando modelos:"
for m in retinaface.onnx arcface_r100.onnx realesr_x4.onnx; do
    if [ -s "$m" ]; then
        echo "  [OK] $m ($(du -h "$m" | cut -f1))"
    else
        echo "  [FAIL] $m"
    fi
done

echo
if command -v trtexec &>/dev/null; then
    echo "[INFO] TensorRT disponible. Ejemplo:"
    echo "trtexec --onnx=modelo.onnx --saveEngine=modelo.trt --fp16 --workspace=2048"
else
    echo "[INFO] TensorRT no disponible. Usando ONNX Runtime."
fi

echo
echo "[DONE] Modelos listos."
