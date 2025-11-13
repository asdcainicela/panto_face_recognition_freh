#!/bin/bash

set -e

echo "=== PANTO Performance Benchmark ==="
echo ""

if [ ! -d "build/bin" ]; then
    echo "ERROR: No hay build. Ejecuta: ./build.sh"
    exit 1
fi

DURATION=30

echo "[1/3] Test: Captura pura (baseline)"
echo "----------------------------------------"
timeout $DURATION ./build/bin/view main 2>&1 | tee /tmp/bench1.log
BASELINE_FPS=$(grep "fps promedio" /tmp/bench1.log | awk '{print $NF}')
echo ""
echo "Captura: ${BASELINE_FPS} FPS"
echo ""

if [ ! -f models/retinaface.onnx ]; then
    echo "WARN: No hay modelo. Test detector omitido."
    exit 0
fi

# Buscar video en orden de prioridad
if [ -f "video1.mp4" ]; then
    VIDEO="video1.mp4"
elif [ -f "video2.mp4" ]; then
    VIDEO="video2.mp4"
else
    VIDEO=$(ls -t *.mp4 2>/dev/null | head -1)
    
    if [ -z "$VIDEO" ]; then
        VIDEO=$(ls -t videos/*.mp4 2>/dev/null | head -1)
    fi
fi

if [ -z "$VIDEO" ]; then
    echo "WARN: No hay video de prueba"
    echo "      Graba uno primero: ./run.sh record 30"
    exit 0
fi

echo "[2/3] Test: Detector con video"
echo "----------------------------------------"
echo "Video: $VIDEO"
echo ""

./build/bin/test_detector_video \
    models/retinaface.onnx \
    "$VIDEO" 0.5 2>&1 | tee /tmp/bench2.log

DETECTOR_MS=$(grep "Tiempo:" /tmp/bench2.log | tail -1 | awk '{print $2}' | tr -d 'ms/frame')
DETECTOR_FPS=$(echo "scale=2; 1000 / $DETECTOR_MS" | bc)

echo ""
echo "Detector: ${DETECTOR_MS}ms/frame (~${DETECTOR_FPS} FPS)"
echo ""

echo "[3/3] Test: GPU Status"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader | head -1
else
    echo "WARN: nvidia-smi no disponible"
fi
echo ""

echo "=== RESUMEN ==="
echo ""
echo "Captura:      ${BASELINE_FPS} FPS"
echo "Detector:     ${DETECTOR_FPS} FPS (${DETECTOR_MS}ms)"
echo ""

TARGET=20
if (( $(echo "$BASELINE_FPS >= $TARGET" | bc -l) )); then
    echo "OK: Captura >= ${TARGET} FPS"
else
    echo "FAIL: Captura < ${TARGET} FPS"
    echo ""
    echo "Posibles causas:"
    echo "  1. GStreamer no usa hardware decode"
    echo "  2. Ejecutando desde SSH sin display"
    echo "  3. CPU/GPU en throttling"
fi

echo ""