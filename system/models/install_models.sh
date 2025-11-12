#!/bin/bash
# Script simple para extraer modelos de buffalo_l.zip

cd /workspace/panto_face_recognition_freh/system/models

echo "=== Extrayendo modelos de buffalo_l.zip ==="
echo ""

# Verificar que existe el ZIP
if [ ! -f "buffalo_l.zip" ]; then
    echo "ERROR: buffalo_l.zip no encontrado en $(pwd)"
    echo ""
    echo "Descarga desde:"
    echo "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    exit 1
fi

echo "✓ buffalo_l.zip encontrado ($(du -h buffalo_l.zip | cut -f1))"
echo ""

# Extraer todo el contenido
echo "Extrayendo archivos..."
unzip -o buffalo_l.zip

echo ""
echo "Archivos extraídos:"
ls -lh buffalo_l/

echo ""
echo "Copiando modelos..."

# Copiar detector
cp buffalo_l/det_10g.onnx retinaface.onnx
echo "✓ det_10g.onnx -> retinaface.onnx"

# Copiar reconocimiento
cp buffalo_l/w600k_r50.onnx arcface_r100.onnx
echo "✓ w600k_r50.onnx -> arcface_r100.onnx"

echo ""
echo "=== RESUMEN ==="
ls -lh retinaface.onnx arcface_r100.onnx

echo ""
echo "✓ Modelos listos!"
echo ""
echo "Probar con:"
echo "  cd .."
echo "  ./build/bin/test_detector models/retinaface.onnx test/img/test1.png"