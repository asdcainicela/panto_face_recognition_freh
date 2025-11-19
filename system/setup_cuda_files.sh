#!/bin/bash

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║        Setting up CUDA files for PANTO system            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 1. CREATE: include/cuda_kernels.h
# ============================================================

echo "Creating include/cuda_kernels.h..."

cat > include/cuda_kernels.h << 'EOF'
// ============= include/cuda_kernels.h =============
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Normalización ImageNet en GPU (BGR -> RGB + normalize)
void cuda_normalize_imagenet(
    const unsigned char* d_input,  // Imagen BGR en GPU
    float* d_output,               // Tensor CHW normalizado en GPU
    int width,                     // Ancho de la imagen
    int height,                    // Alto de la imagen
    cudaStream_t stream            // Stream CUDA para async
);

#ifdef __cplusplus
}
#endif
EOF

echo "  ✓ include/cuda_kernels.h created"

# ============================================================
# 2. CREATE: src/cuda_kernels.cu
# ============================================================

echo "Creating src/cuda_kernels.cu..."

cat > src/cuda_kernels.cu << 'EOF'
// ============= src/cuda_kernels.cu =============
#include <cuda_runtime.h>

// Kernel optimizado para normalización ImageNet (BGR -> RGB + normalize)
__global__ void normalize_imagenet_kernel(
    const unsigned char* input,  // BGR image
    float* output,               // CHW float tensor
    int width, int height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixel_idx = idx * 3;  // BGR format
    
    // BGR input
    float b = input[pixel_idx + 0] / 255.0f;
    float g = input[pixel_idx + 1] / 255.0f;
    float r = input[pixel_idx + 2] / 255.0f;
    
    // Output: CHW format, RGB order, normalized
    int plane_size = width * height;
    output[0 * plane_size + idx] = (r - mean_r) / std_r;  // R channel
    output[1 * plane_size + idx] = (g - mean_g) / std_g;  // G channel
    output[2 * plane_size + idx] = (b - mean_b) / std_b;  // B channel
}

// Wrapper C++ para el kernel
extern "C" void cuda_normalize_imagenet(
    const unsigned char* d_input, 
    float* d_output,
    int width, 
    int height, 
    cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    const float mean_r = 0.485f, mean_g = 0.456f, mean_b = 0.406f;
    const float std_r = 0.229f, std_g = 0.224f, std_b = 0.225f;
    
    normalize_imagenet_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        mean_r, mean_g, mean_b, std_r, std_g, std_b
    );
}
EOF

echo "  ✓ src/cuda_kernels.cu created"

# ============================================================
# 3. BACKUP & UPDATE: src/detector_optimized.cpp
# ============================================================

echo "Updating src/detector_optimized.cpp..."

if [ -f "src/detector_optimized.cpp" ]; then
    cp src/detector_optimized.cpp src/detector_optimized.cpp.backup
    echo "  ✓ Backup created: src/detector_optimized.cpp.backup"
fi

# Verificar si ya tiene el include correcto
if grep -q '#include "cuda_kernels.h"' src/detector_optimized.cpp; then
    echo "  ✓ src/detector_optimized.cpp already includes cuda_kernels.h"
else
    # Agregar include después de detector_optimized.hpp
    sed -i '/#include "detector_optimized.hpp"/a #include "cuda_kernels.h"' src/detector_optimized.cpp
    echo "  ✓ Added #include \"cuda_kernels.h\" to detector_optimized.cpp"
fi

# Remover la declaración del wrapper si existe (evitar duplicados)
if grep -q "^void cuda_normalize_imagenet" src/detector_optimized.cpp; then
    sed -i '/^void cuda_normalize_imagenet/,/^}/d' src/detector_optimized.cpp
    echo "  ✓ Removed duplicate cuda_normalize_imagenet declaration"
fi

# Remover kernels CUDA si existen
if grep -q "__global__" src/detector_optimized.cpp; then
    sed -i '/__global__/,/^}/d' src/detector_optimized.cpp
    echo "  ✓ Removed CUDA kernels from .cpp file"
fi

echo "  ✓ src/detector_optimized.cpp updated"

# ============================================================
# 4. BACKUP & UPDATE: include/detector_optimized.hpp
# ============================================================

echo "Checking include/detector_optimized.hpp..."

if [ -f "include/detector_optimized.hpp" ]; then
    # Remover declaración del wrapper si existe
    if grep -q "^void cuda_normalize_imagenet" include/detector_optimized.hpp; then
        cp include/detector_optimized.hpp include/detector_optimized.hpp.backup
        sed -i '/^void cuda_normalize_imagenet/,/^;/d' include/detector_optimized.hpp
        echo "  ✓ Removed cuda_normalize_imagenet declaration from header"
    else
        echo "  ✓ Header is already clean"
    fi
fi

# ============================================================
# 5. VERIFICATION
# ============================================================

echo ""
echo "=== Verification ==="
echo ""

FILES_CREATED=(
    "include/cuda_kernels.h"
    "src/cuda_kernels.cu"
)

ALL_OK=1
for file in "${FILES_CREATED[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(wc -l < "$file")
        echo "  ✓ $file ($SIZE lines)"
    else
        echo "  ✗ FAILED: $file"
        ALL_OK=0
    fi
done

echo ""

if [ $ALL_OK -eq 1 ]; then
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           CUDA FILES SETUP SUCCESSFUL! ✓                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Files created:"
    echo "  - include/cuda_kernels.h    (CUDA kernel declarations)"
    echo "  - src/cuda_kernels.cu       (CUDA kernel implementations)"
    echo ""
    echo "Files updated:"
    echo "  - src/detector_optimized.cpp (includes cuda_kernels.h)"
    echo ""
    echo "Backups created (if needed):"
    echo "  - src/detector_optimized.cpp.backup"
    echo "  - include/detector_optimized.hpp.backup"
    echo ""
    echo "Next step: Run clean build"
    echo "  ./clean_build.sh"
    echo ""
else
    echo "ERROR: Some files failed to create!"
    exit 1
fi