#!/bin/bash
set -e

CORES=6
BUILD_TYPE="${1:-Release}"

echo "╔════════════════════════════════════════╗"
echo "║         Building PANTO System          ║"
echo "╚════════════════════════════════════════╝"
echo "Build type: $BUILD_TYPE"
echo "Cores: $CORES (Jetson Orin)"
echo ""

mkdir -p build
cd build

if [ ! -f "Makefile" ]; then
    echo "=== Configuring CMake ==="
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
else
    echo "=== Using existing configuration ==="
fi

echo "=== Building (incremental) ==="
make -j$CORES

echo ""
echo "╔════════════════════════════════════════╗"
echo "║          Build Complete                ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Shared Libraries:"
echo "  ✓ libpanto_utils.so        (utilities)"
echo "  ✓ libpanto_draw.so         (drawing)"
echo "  ✓ libpanto_stream.so       (RTSP capture)"
echo "  ✓ libpanto_detector_opt.so (TensorRT + CUDA)"
echo "  ✓ libpanto_cuda_kernels.a  (CUDA kernels)"
echo ""
echo "Executables:"
echo "  ✓ panto               (main app - face detection)"
echo "  ✓ benchmark_detector  (performance test)"
echo "  ✓ view                (view streams only)"
echo "  ✓ record              (record streams only)"
echo ""
echo "Quick Start:"
echo "  ./build/bin/panto config.toml"
echo "  ./build/bin/benchmark_detector models/scrfd.engine image.jpg"
echo ""

# Verificación
MISSING=0

if [ ! -f "lib/libpanto_detector_opt.so" ]; then
    echo "✗ ERROR: libpanto_detector_opt.so not found!"
    MISSING=1
fi

if [ ! -f "lib/libpanto_cuda_kernels.a" ]; then
    echo "✗ ERROR: libpanto_cuda_kernels.a not found!"
    MISSING=1
fi

if [ ! -f "bin/panto" ]; then
    echo "✗ ERROR: panto not found!"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  WARNING: Build incomplete!"
    exit 1
else
    echo "✓ All components built successfully!"
fi