#!/bin/bash

set -e

echo "=== Building PANTO ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j4

echo ""
echo "=== Build Complete ==="
echo ""
echo "Executables in: build/bin/"
echo "  - record"
echo "  - view"
echo "  - record_view"
echo "  - panto"
echo ""
echo "Libraries in: build/lib/"
echo "  - libstream_lib.so"
echo ""