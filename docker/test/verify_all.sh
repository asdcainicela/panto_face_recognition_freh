#!/bin/bash

# Colors
RESET='\033[0m'
BOLD='\033[1m'
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'

print_header() {
    echo -e "\n${BOLD}${CYAN}╔$(printf '═%.0s' {1..60})╗"
    printf "${BOLD}${CYAN}║ %-58s ║\n" "$1"
    echo -e "╚$(printf '═%.0s' {1..60})╝${RESET}\n"
}

print_section() {
    echo -e "${BOLD}${YELLOW}[$1]${RESET}"
}

print_success() {
    echo -e "${GREEN}  ✓ $1${RESET}"
}

print_error() {
    echo -e "${RED}  ✗ $1${RESET}"
}

print_info() {
    echo -e "  • ${BOLD}$1${RESET}: $2"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${RESET}"
}

print_header "VERIFICACIÓN COMPLETA DEL SISTEMA"

# ===== 1. INFORMACIÓN DEL SISTEMA =====
print_section "1. Información del Sistema"
print_info "Hostname" "$(hostname)"
print_info "Kernel" "$(uname -r)"
print_info "Architecture" "$(uname -m)"
print_info "Date" "$(date '+%Y-%m-%d %H:%M:%S')"
print_info "Timezone" "$(cat /etc/timezone 2>/dev/null || echo 'Unknown')"
echo ""

# ===== 2. COMPILADORES =====
print_section "2. Compiladores"
if command -v gcc &> /dev/null; then
    gcc_ver=$(gcc --version | head -n1)
    print_info "GCC" "$gcc_ver"
    
    # Check alternatives
    print_info "GCC Active" "$(gcc --version | head -n1 | awk '{print $3}')"
    
    if command -v gcc-11 &> /dev/null; then
        print_success "GCC 11 disponible"
    fi
    if command -v gcc-9 &> /dev/null; then
        print_success "GCC 9 disponible"
    fi
else
    print_error "GCC no encontrado"
fi

if command -v g++ &> /dev/null; then
    gxx_ver=$(g++ --version | head -n1)
    print_info "G++" "$gxx_ver"
else
    print_error "G++ no encontrado"
fi

if command -v cmake &> /dev/null; then
    cmake_ver=$(cmake --version | head -n1)
    print_info "CMake" "$cmake_ver"
else
    print_error "CMake no encontrado"
fi
echo ""

# ===== 3. CUDA & GPU =====
print_section "3. CUDA & GPU"

if command -v nvcc &> /dev/null; then
    cuda_ver=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_info "CUDA Version" "$cuda_ver"
    print_info "NVCC Path" "$(which nvcc)"
    print_success "CUDA Toolkit instalado"
else
    print_error "CUDA Toolkit no encontrado"
fi

if command -v nvidia-smi &> /dev/null; then
    echo ""
    print_info "GPU Information" ""
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | while read line; do
        echo "    $line"
    done
    print_success "nvidia-smi disponible"
else
    print_warning "nvidia-smi no disponible (normal en algunos Jetson)"
fi

# Check CUDA libraries
if [ -d "/usr/local/cuda/lib64" ]; then
    print_success "CUDA libraries encontradas en /usr/local/cuda/lib64"
fi
echo ""

# ===== 4. OPENCV =====
print_section "4. OpenCV"

if pkg-config --exists opencv4; then
    opencv_ver=$(pkg-config --modversion opencv4)
    print_info "Versión" "$opencv_ver"
    print_success "OpenCV 4 instalado"
    
    # Check for CUDA support in OpenCV
    opencv_build=$(pkg-config --variable=prefix opencv4)/lib/cmake/opencv4
    if [ -f "${opencv_build}/OpenCVModules.cmake" ]; then
        if grep -q "cuda" "${opencv_build}/OpenCVModules.cmake" 2>/dev/null; then
            print_success "OpenCV compilado con CUDA"
        else
            print_warning "OpenCV sin soporte CUDA"
        fi
    fi
    
    # Check modules
    opencv_libs=$(pkg-config --libs opencv4)
    if echo "$opencv_libs" | grep -q "cudaarithm"; then
        print_success "Módulos CUDA detectados"
    fi
    
elif command -v opencv_version &> /dev/null; then
    opencv_ver=$(opencv_version)
    print_info "Versión" "$opencv_ver"
    print_warning "pkg-config no configurado, pero OpenCV está instalado"
else
    print_error "OpenCV no encontrado"
fi
echo ""

# ===== 5. TENSORRT =====
print_section "5. TensorRT"

tensorrt_lib="/usr/lib/aarch64-linux-gnu/libnvinfer.so"
tensorrt_header="/usr/include/aarch64-linux-gnu/NvInfer.h"

if [ -f "$tensorrt_lib" ]; then
    tensorrt_ver=$(ls -l $tensorrt_lib* | head -n1 | awk -F'.' '{print $(NF-2)"."$(NF-1)"."$NF}' | sed 's/so.//')
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
    python_ver=$(python3 --version | awk '{print $2}')
    print_info "Versión" "$python_ver"
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
        print(f"    • {pkg:15} : {version}")
    except ImportError:
        print(f"    ✗ {pkg:15} : No instalado")
EOF
echo ""

# ===== 7. JUPYTER =====
print_section "7. JupyterLab"

if command -v jupyter &> /dev/null; then
    jupyter_ver=$(jupyter --version | grep "jupyterlab" | awk '{print $3}')
    print_info "Versión" "$jupyter_ver"
    print_info "URL" "http://localhost:8888"
    print_info "Token" "nvidia"
    
    if pgrep -f "jupyter-lab" > /dev/null; then
        print_success "JupyterLab corriendo"
    else
        print_warning "JupyterLab no está corriendo"
    fi
    
    if [ -f "/var/log/jupyter.log" ]; then
        print_info "Log" "/var/log/jupyter.log"
    fi
else
    print_error "JupyterLab no instalado"
fi
echo ""

# ===== 8. COMPILAR Y EJECUTAR TESTS C++ =====
print_section "8. Tests C++"

cd /opt/tests

if [ -f "CMakeLists.txt" ] && [ -f "test_all.cpp" ]; then
    echo ""
    echo "  Compilando tests C++..."
    
    rm -rf build
    mkdir -p build && cd build
    
    if cmake .. > /tmp/cmake_output.log 2>&1; then
        print_success "CMake configuration exitosa"
        
        if make -j$(nproc) > /tmp/make_output.log 2>&1; then
            print_success "Compilación exitosa"
            
            if [ -f "test_all" ]; then
                echo ""
                print_info "Ejecutando" "test_all"
                echo ""
                echo "----------------------------------------"
                ./test_all
                echo "----------------------------------------"
            else
                print_error "Ejecutable test_all no encontrado"
            fi
        else
            print_error "Error en compilación"
            echo "  Ver detalles en: /tmp/make_output.log"
        fi
    else
        print_error "Error en CMake configuration"
        echo "  Ver detalles en: /tmp/cmake_output.log"
    fi
else
    print_error "CMakeLists.txt o test_all.cpp no encontrados"
fi
echo ""

cd /opt/tests

# ===== 9. TESTS PYTHON =====
print_section "9. Tests Python"

if [ -f "test_all.py" ]; then
    echo ""
    python3 test_all.py
else
    print_error "test_all.py no encontrado"
fi
echo ""

# ===== 10. VERIFICACIONES ADICIONALES =====
print_section "10. Verificaciones Adicionales"

# Check display
if [ ! -z "$DISPLAY" ]; then
    print_success "DISPLAY configurado: $DISPLAY"
else
    print_warning "DISPLAY no configurado"
fi

# Check X11
if [ -d "/tmp/.X11-unix" ]; then
    print_success "Socket X11 disponible"
else
    print_warning "Socket X11 no encontrado"
fi

# Check devices
if [ -e "/dev/video0" ]; then
    print_success "Cámara /dev/video0 detectada"
else
    print_warning "Cámara /dev/video0 no encontrada"
fi

# Check memory
total_mem=$(free -g | awk '/^Mem:/{print $2}')
print_info "RAM Total" "${total_mem}GB"

# Check disk space
disk_space=$(df -h /workspace | tail -1 | awk '{print $4}')
print_info "Espacio disponible /workspace" "$disk_space"

echo ""

# ===== RESUMEN FINAL =====
print_header "RESUMEN FINAL"

echo -e "${BOLD}  Componentes Verificados:${RESET}"
echo ""

# Create summary
declare -A status
status["GCC"]=$(command -v gcc &> /dev/null && echo "✓" || echo "✗")
status["CMake"]=$(command -v cmake &> /dev/null && echo "✓" || echo "✗")
status["CUDA"]=$(command -v nvcc &> /dev/null && echo "✓" || echo "✗")
status["OpenCV"]=$(pkg-config --exists opencv4 && echo "✓" || echo "✗")
status["TensorRT"]=$([ -f "/usr/lib/aarch64-linux-gnu/libnvinfer.so" ] && echo "✓" || echo "✗")
status["Python"]=$(command -v python3 &> /dev/null && echo "✓" || echo "✗")
status["JupyterLab"]=$(command -v jupyter &> /dev/null && echo "✓" || echo "✗")

for key in "${!status[@]}"; do
    if [ "${status[$key]}" = "✓" ]; then
        echo -e "  ${GREEN}✓${RESET} $key"
    else
        echo -e "  ${RED}✗${RESET} $key"
    fi
done

echo ""
echo -e "${BOLD}${CYAN}  Verificación completa finalizada${RESET}"
echo ""