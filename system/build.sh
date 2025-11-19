#!/bin/bash

set -e

# Usar 6 cores para Jetson Orin
CORES=6
BUILD_TYPE="${1:-Release}"

echo "=== Building PANTO ==="
echo "Build type: $BUILD_TYPE"
echo "Using $CORES cores (fixed for Jetson Orin stability)"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with CMake (solo si es necesario)
if [ ! -f "Makefile" ]; then
    echo "=== Configuring with CMake ==="
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
else
    echo "=== Using existing CMake configuration ==="
fi

# Build (incremental - solo recompila lo que cambió)
echo "=== Building (incremental) ==="
make -j$CORES

echo ""
echo "=== Build Complete ==="
echo ""
echo "Modular shared libraries:"
echo "  - build/lib/libpanto_utils.so          (base utilities)"
echo "  - build/lib/libpanto_draw.so           (drawing utilities)"
echo "  - build/lib/libpanto_stream.so         (RTSP capture)"
echo "  - build/lib/libpanto_detector.so       (TensorRT detector - original)"
echo "  - build/lib/libpanto_detector_opt.so   (TensorRT detector - OPTIMIZED)"
echo "  - build/lib/libpanto_cuda_kernels.a    (CUDA kernels)"
echo ""
echo "Executables:"
echo "  - build/bin/record                 (headless recording)"
echo "  - build/bin/view                   (viewing only)"
echo "  - build/bin/record_view            (record + view)"
echo "  - build/bin/panto                  (main app)"
echo "  - build/bin/test_detector          (test detector on image/webcam)"
echo "  - build/bin/test_detector_video    (test detector on video)"
echo "  - build/bin/benchmark_detector     (benchmark original vs optimized)"
echo "  - build/bin/test_detector_video_opt (test optimized detector)"
echo ""
echo "Quick commands:"
echo "  ./run.sh 1     # Record main stream"
echo "  ./run.sh 3     # View both streams"
echo "  ./run.sh 8     # Run PANTO"
echo ""
echo "Benchmark & Test:"
echo "  ./build/bin/benchmark_detector models/scrfd_10g_bnkps.engine test_image.jpg"
echo "  ./build/bin/test_detector_video models/scrfd_10g_bnkps.engine video.mp4"
echo ""

# Verificar que los binarios importantes se compilaron
echo "=== Verification ==="
MISSING=0

if [ -f "lib/libpanto_detector_opt.so" ]; then
    echo "✓ Optimized detector compiled successfully"
else
    echo "✗ ERROR: libpanto_detector_opt.so not found!"
    MISSING=1
fi

if [ -f "lib/libpanto_cuda_kernels.a" ]; then
    echo "✓ CUDA kernels compiled successfully"
else
    echo "✗ ERROR: libpanto_cuda_kernels.a not found!"
    MISSING=1
fi

if [ -f "bin/benchmark_detector" ]; then
    echo "✓ Benchmark tool compiled successfully"
else
    echo "✗ ERROR: benchmark_detector not found!"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  WARNING: Some components failed to compile!"
    echo "Check the build output above for errors."
    exit 1
else
    echo ""
    echo "✓ All components compiled successfully!"
fi