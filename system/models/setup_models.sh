#!/bin/bash
set -e

CONFIG_FILE="../config.toml"
MODEL_DIR="."

echo "[INFO] Descargando modelos ONNX..."
wget -q -N https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/model/retinaface.onnx
wget -q -N https://github.com/deepinsight/insightface/releases/download/v1.0/arcface_r100.onnx
wget -q -N https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/realesr_x4.onnx

echo "[INFO] Convirtiendo modelos a TensorRT..."
convert_model() {
    local model=$1
    local output="${model%.onnx}.trt"
    if command -v trtexec &>/dev/null; then
        trtexec --onnx="$model" --saveEngine="$output" --fp16 --workspace=2048 &>/dev/null && echo "[OK] $output generado"
    else
        echo "[WARN] TensorRT no encontrado, omitiendo conversión de $model"
    fi
}

convert_model "retinaface.onnx"
convert_model "arcface_r100.onnx"
convert_model "realesr_x4.onnx"

echo "[INFO] Verificando resultados..."
if ls *.trt &>/dev/null; then
    echo "[INFO] Motores TensorRT detectados. Actualizando config.toml..."
    sed -i 's/use_tensorrt *= *.*/use_tensorrt = true/' "$CONFIG_FILE" 2>/dev/null || true
else
    echo "[INFO] No se generaron motores TensorRT. Desactivando en config.toml..."
    sed -i 's/use_tensorrt *= *.*/use_tensorrt = false/' "$CONFIG_FILE" 2>/dev/null || true
fi

echo "[INFO] Modelos disponibles:"
ls -lh *.onnx *.trt 2>/dev/null || true

echo "[DONE] Configuración de modelos completada."
