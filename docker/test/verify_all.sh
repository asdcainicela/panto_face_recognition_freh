#!/bin/bash

print_header() {
    echo ""
    echo "$1"
    echo ""
}

print_section() {
    echo "[$1]"
}

print_success() {
    echo "  OK: $1"
}

print_error() {
    echo "  ERROR: $1"
}

print_info() {
    echo "  - $1: $2"
}

print_warning() {
    echo "  WARNING: $1"
}

print_header "VERIFICACION COMPLETA DEL SISTEMA"

# ===== 1. INFORMACION DEL SISTEMA =====
print_section "1. Informacion del Sistema"
print_info "Hostname" "$(hostname)"
print_info "Kernel" "$(uname -r)"
print_info "Architecture" "$(uname -m)"
print_info "Date" "$(date '+%Y-%m-%d %H:%M:%S')"
print_info "Timezone" "$(cat /etc/timezone 2>/dev/null || echo 'Unknown')"
echo ""

# ===== 2. COMPILADORES =====
print_section "2. Compiladores"
if command -v gcc &> /dev/null; then
    print_info "GCC" "$(gcc --version | head -n1)"
    print_info "GCC Active" "$(gcc --version | head -n1 | awk '{print $3}')"
    command -v gcc-11 &> /dev/null && print_success "GCC 11 disponible"
    command -v gcc-9 &> /dev/null && print_success "GCC 9 disponible"
else
    print_error "GCC no encontrado"
fi

if command -v g++ &> /dev/null; then
    print_info "G++" "$(g++ --version | head -n1)"
else
    print_error "G++ no encontrado"
fi

if command -v cmake &> /dev/null; then
    print_info "CMake" "$(cmake --version | head -n1)"
else
    print_error "CMake no encontrado"
fi
echo ""

# ===== 3. CUDA & GPU =====
print_section "3. CUDA y GPU"

if command -v nvcc &> /dev/null; then
    cuda_ver=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_info "CUDA Version" "$cuda_ver"
    print_info "NVCC Path" "$(which nvcc)"
    print_success "CUDA Toolkit instalado"
else
    print_error "CUDA Toolkit no encontrado"
fi

if command -v nvidia-smi &> /dev/null; then
    print_info "GPU Information" ""
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    print_success "nvidia-smi disponible"
else
    print_warning "nvidia-smi no disponible en este dispositivo"
fi

if [ -d "/usr/local/cuda/lib64" ]; then
    print_success "CUDA libraries encontradas en /usr/local/cuda/lib64"
fi
echo ""

# ===== 4. OPENCV =====
print_section "4. OpenCV"

if pkg-config --exists opencv4; then
    print_info "Version" "$(pkg-config --modversion opencv4)"
    print_success "OpenCV 4 instalado"

    opencv_build=$(pkg-config --variable=prefix opencv4)/lib/cmake/opencv4
    if [ -f "${opencv_build}/OpenCVModules.cmake" ]; then
        if grep -q "cuda" "${opencv_build}/OpenCVModules.cmake" 2>/dev/null; then
            print_success "OpenCV compilado con CUDA"
        else
            print_warning "OpenCV sin soporte CUDA"
        fi
    fi

    if pkg-config --libs opencv4 | grep -q "cudaarithm"; then
        print_success "Modulos CUDA detectados"
    fi

elif command -v opencv_version &> /dev/null; then
    print_info "Version" "$(opencv_version)"
    print_warning "OpenCV instalado pero pkg-config no configurado"
else
    print_error "OpenCV no encontrado"
fi
echo ""

# ===== 5. TENSORRT =====
print_section "5. TensorRT"

tensorrt_lib="/usr/lib/aarch64-linux-gnu/libnvinfer.so"
tensorrt_header="/usr/include/aarch64-linux-gnu/NvInfer.h"

if [ -f "$tensorrt_lib" ]; then
    print_info "Library" "$tensorrt_lib"
    print_success "TensorRT library encontrada"
else
    print_error "TensorRT library no encontrada"
fi

if [ -f "$tensorrt_header" ]; then
    print_info "Header" "$tensorrt_header"
    print_success "TensorRT headers encontrados"
else
    print_error "TensorRT headers no encontrados"
fi
echo ""

# ===== 6. PYTHON =====
print_section "6. Python"

if command -v python3 &> /dev/null; then
    print_info "Version" "$(python3 --version | awk '{print $2}')"
    print_info "Path" "$(which python3)"
    print_success "Python 3 instalado"
else
    print_error "Python 3 no encontrado"
fi

echo ""
print_info "Paquetes Python" ""
python3 << 'EOF'
packages = [
    'numpy', 'cv2', 'matplotlib', 'pandas',
    'scipy', 'sklearn', 'seaborn', 'PIL', 'tqdm',
    'jupyterlab'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'OK')
        print(f"    {pkg:15} : {version}")
    except ImportError:
        print(f"    {pkg:15} : No instalado")
EOF
echo ""

# ===== 7. JUPYTER =====
print_section "7. JupyterLab"

if command -v jupyter &> /dev/null; then
    print_info "Version" "$(jupyter --version | grep jupyterlab | awk '{print $3}')"
    print_info "URL" "http://localhost:8888"
    print_info "Token" "nvidia"

    if pgrep -f "jupyter-lab" > /dev/null; then
        print_success "JupyterLab corriendo"
    else
        print_warning "JupyterLab no esta corriendo"
    fi

    if [ -f "/var/log/jupyter.log" ]; then
        print_info "Log" "/var/log/jupyter.log"
    fi
else
    print_error "JupyterLab no instalado"
fi
echo ""

# ===== 8. TESTS C++ =====
print_section "8. Tests C++"

cd /opt/tests

if [ -f "CMakeLists.txt" ] && [ -f "test_all.cpp" ]; then
    echo "  Compilando tests C++..."

    rm -rf build
    mkdir -p build && cd build

    if cmake .. > /tmp/cmake_output.log 2>&1; then
        print_success "CMake configuration exitosa"

        if make -j$(nproc) > /tmp/make_output.log 2>&1; then
            print_success "Compilacion exitosa"

            if [ -f "test_all" ]; then
                print_info "Ejecutando" "test_all"
                echo "----------------------------------------"
                ./test_all
                echo "----------------------------------------"
            else
                print_error "Ejecutable test_all no encontrado"
            fi
        else
            print_error "Error en compilacion"
            echo "  Revisar /tmp/make_output.log"
        fi
    else
        print_error "Error en configuracion CMake"
        echo "  Revisar /tmp/cmake_output.log"
    fi
else
    print_error "CMakeLists.txt o test_all.cpp no encontrados"
fi
echo ""

cd /opt/tests

# ===== 9. TESTS PYTHON =====
print_section "9. Tests Python"

if [ -f "test_all.py" ]; then
    python3 test_all.py
else
    print_error "test_all.py no encontrado"
fi
echo ""

# ===== 10. VERIFICACIONES ADICIONALES =====
print_section "10. Verificaciones Adicionales"

[ ! -z "$DISPLAY" ] && print_success "DISPLAY configurado: $DISPLAY" || print_warning "DISPLAY no configurado"
[ -d "/tmp/.X11-unix" ] && print_success "Socket X11 disponible" || print_warning "Socket X11 no encontrado"
[ -e "/dev/video0" ] && print_success "Camara /dev/video0 detectada" || print_warning "Camara /dev/video0 no encontrada"

total_mem=$(free -g | awk '/^Mem:/{print $2}')
print_info "RAM Total" "${total_mem}GB"

disk_space=$(df -h /workspace | tail -1 | awk '{print $4}')
print_info "Espacio disponible en /workspace" "$disk_space"

echo ""

# ===== RESUMEN FINAL =====
print_header "RESUMEN FINAL"

declare -A status
status["GCC"]=$(command -v gcc &> /dev/null && echo "OK" || echo "NO")
status["CMake"]=$(command -v cmake &> /dev/null && echo "OK" || echo "NO")
status["CUDA"]=$(command -v nvcc &> /dev/null && echo "OK" || echo "NO")
status["OpenCV"]=$(pkg-config --exists opencv4 && echo "OK" || echo "NO")
status["TensorRT"]=$([ -f "/usr/lib/aarch64-linux-gnu/libnvinfer.so" ] && echo "OK" || echo "NO")
status["Python"]=$(command -v python3 &> /dev/null && echo "OK" || echo "NO")
status["JupyterLab"]=$(command -v jupyter &> /dev/null && echo "OK" || echo "NO")

for key in "${!status[@]}"; do
    echo "  $key: ${status[$key]}"
done

echo ""
echo "Verificacion completa finalizada"
echo ""
