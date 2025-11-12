#!/bin/bash

set -e

CORES=$(nproc)
BUILD_TYPE="${1:-Release}"

echo "=== Building PANTO ==="
echo "Build type: $BUILD_TYPE"
echo "Using $CORES cores"
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
echo "  - build/lib/libpanto_utils.so   (base)"
echo "  - build/lib/libpanto_draw.so    (drawing)"
echo "  - build/lib/libpanto_stream.so  (capture)"
echo ""
echo "Executables:"
echo "  - build/bin/record       (headless recording)"
echo "  - build/bin/view         (viewing only)"
echo "  - build/bin/record_view  (record + view)"
echo "  - build/bin/panto        (main app)"
echo ""
echo "Quick commands:"
echo "  ./run.sh 1     # Record main stream"
echo "  ./run.sh 3     # View both streams"
echo "  ./run.sh 8     # Run PANTO"
echo ""