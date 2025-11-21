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

echo "=== Building ==="
make -j$CORES

echo ""
echo "╔════════════════════════════════════════╗"
echo "║          Build Complete                ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Executables:"
echo "  ✓ panto       (main app)"
echo "  ✓ benchmark   (performance test)"
echo ""
echo "Ejemplos de uso:"
echo ""
echo "  # Solo captura (sin detector)"
echo "  ./build/bin/panto config.toml"
echo "  # Editar config.toml: detect=false, record=false, display=true"
echo ""
echo "  # Solo grabar (sin display)"
echo "  ./build/bin/panto config.toml"
echo "  # Editar config.toml: detect=false, record=true, display=false"
echo ""
echo "  # Detección + Display"
echo "  ./build/bin/panto config.toml"
echo "  # Editar config.toml: detect=true, record=false, display=true"
echo ""
echo "  # Detección + Grabar + Display (TODO)"
echo "  ./build/bin/panto config.toml"
echo "  # Editar config.toml: detect=true, record=true, display=true"
echo ""
echo "  # Benchmark"
echo "  ./build/bin/benchmark models/scrfd.engine imagen.jpg"
echo ""

# Verificación
MISSING=0

if [ ! -f "bin/panto" ]; then
    echo "✗ ERROR: panto no compilado"
    MISSING=1
fi

if [ ! -f "bin/benchmark" ]; then
    echo "✗ ERROR: benchmark no compilado"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Build incompleto!"
    exit 1
else
    echo "✓ Build exitoso!"
fi