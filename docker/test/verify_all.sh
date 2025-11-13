#!/bin/bash

echo ""
echo "System Verification Starting"
echo ""

echo "System Info"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Date: $(date)"
echo ""

echo "Compilers"
gcc --version | head -n1
g++ --version | head -n1
cmake --version | head -n1
echo ""

echo "CUDA & GPU"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep release
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi
else
    echo "CUDA: not found"
fi
echo ""

echo "OpenCV"
if pkg-config --exists opencv4; then
    pkg-config --modversion opencv4
else
    echo "OpenCV: not found"
fi
echo ""

echo "ONNX Runtime"
if [ -d "/opt/onnxruntime" ]; then
    ls /opt/onnxruntime/lib/libonnxruntime.so* 2>/dev/null | head -n1
else
    echo "ONNX: not found"
fi
echo ""

echo "Python Packages"
python3 -c "
packages = ['numpy', 'onnxruntime', 'matplotlib', 'pandas', 'scipy', 'sklearn']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'OK')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: not found')
"
echo ""

cd /opt/tests

echo "Building C++ Tests"
mkdir -p build && cd build
cmake .. > /dev/null 2>&1
if make -j4 > /dev/null 2>&1; then
    echo "Build: OK"
    echo ""
    ./test_all
else
    echo "Build: FAIL"
fi
echo ""

cd /opt/tests

echo "Running Python Tests"
python3 test_all.py
echo ""

echo "Verification Complete"
echo ""