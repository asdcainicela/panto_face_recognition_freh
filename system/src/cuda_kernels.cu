// ============= src/cuda_kernels.cu - VERSION CON DEBUG =============

#include <cuda_runtime.h>
#include <cstdio>

// ==================== KERNEL SCRFD - CON VALIDACIÓN ====================
__global__ void normalize_imagenet_kernel(
    const unsigned char* input,
    float* output,
    int width, int height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixel_idx = idx * 3;  // BGR format
    
    // ✅ Leer valores raw [0-255]
    unsigned char b_raw = input[pixel_idx + 0];
    unsigned char g_raw = input[pixel_idx + 1];
    unsigned char r_raw = input[pixel_idx + 2];
    
    // Convertir a float
    float b = static_cast<float>(b_raw);
    float g = static_cast<float>(g_raw);
    float r = static_cast<float>(r_raw);
    
    // ✅ DEBUG: Primer pixel (solo thread 0,0)
    /*if (x == 0 && y == 0) {
        printf("🔍 [CUDA Kernel] Primer pixel BGR: (%u, %u, %u)\n", 
               b_raw, g_raw, r_raw);
        printf("🔍 [CUDA Kernel] Normalizado antes: (%.2f, %.2f, %.2f)\n", 
               b, g, r);
    }*/
    
    // ✅ Normalizar: (pixel - mean) / std
    float r_norm = (r - mean_r) / std_r;
    float g_norm = (g - mean_g) / std_g;
    float b_norm = (b - mean_b) / std_b;
    
    // ✅ DEBUG: Verificar rangos
    /*
    if (x == 0 && y == 0) {
        printf("🔍 [CUDA Kernel] Normalizado después: (%.4f, %.4f, %.4f)\n", 
               r_norm, g_norm, b_norm);
        printf("🔍 [CUDA Kernel] Rangos esperados: [-1.0, +1.0]\n");
    }
    */
   
    // ✅ Escribir en formato CHW (Channel, Height, Width)
    int plane_size = width * height;
    output[0 * plane_size + idx] = r_norm;  // R channel
    output[1 * plane_size + idx] = g_norm;  // G channel
    output[2 * plane_size + idx] = b_norm;  // B channel
    
    // ✅ VALIDACIÓN: Detectar valores anormales
    if (r_norm < -10.0f || r_norm > 10.0f || 
        g_norm < -10.0f || g_norm > 10.0f || 
        b_norm < -10.0f || b_norm > 10.0f) {
        printf("⚠️ [CUDA Kernel] Valor anormal en (%d,%d): R=%.2f G=%.2f B=%.2f\n",
               x, y, r_norm, g_norm, b_norm);
    }
}

extern "C" void cuda_normalize_imagenet(
    const unsigned char* d_input, 
    float* d_output,
    int width, 
    int height, 
    cudaStream_t stream) 
{
    // ✅ Validar punteros
    if (d_input == nullptr || d_output == nullptr) {
        printf("❌ [CUDA] Null pointer: input=%p output=%p\n", d_input, d_output);
        return;
    }
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    // ✅ SCRFD normalization: (pixel - 127.5) / 128.0
    const float mean_r = 127.5f;
    const float mean_g = 127.5f;
    const float mean_b = 127.5f;
    const float std_r = 128.0f;
    const float std_g = 128.0f;
    const float std_b = 128.0f;
    
    printf("🚀 [CUDA] Lanzando kernel: %dx%d grid=(%d,%d) block=(%d,%d)\n",
           width, height, grid.x, grid.y, block.x, block.y);
    
    normalize_imagenet_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        mean_r, mean_g, mean_b, std_r, std_g, std_b
    );
    
    // ✅ Verificar errores de lanzamiento
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("❌ [CUDA] Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return;
    }
    
    // ✅ Sincronizar para capturar errores de ejecución
    cudaError_t exec_err = cudaStreamSynchronize(stream);
    if (exec_err != cudaSuccess) {
        printf("❌ [CUDA] Kernel execution error: %s\n", cudaGetErrorString(exec_err));
    } else {
        printf("✅ [CUDA] Kernel completado exitosamente\n");
    }
}

// ==================== KERNEL AGE/GENDER (sin cambios) ====================
__global__ void normalize_age_gender_kernel(
    const unsigned char* input,
    float* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int pixel_idx = idx * 3;
    
    // BGR -> RGB + normalize to [0, 1] primero
    float b = input[pixel_idx + 0] / 255.0f;
    float g = input[pixel_idx + 1] / 255.0f;
    float r = input[pixel_idx + 2] / 255.0f;
    
    // Luego aplicar ImageNet normalization
    const float mean_r = 0.485f;
    const float mean_g = 0.456f;
    const float mean_b = 0.406f;
    const float std_r = 0.229f;
    const float std_g = 0.224f;
    const float std_b = 0.225f;
    
    // CHW format, normalized
    int plane_size = width * height;
    output[0 * plane_size + idx] = (r - mean_r) / std_r;  // R
    output[1 * plane_size + idx] = (g - mean_g) / std_g;  // G
    output[2 * plane_size + idx] = (b - mean_b) / std_b;  // B
}

extern "C" void cuda_normalize_age_gender(
    const unsigned char* d_input, 
    float* d_output,
    int width, 
    int height, 
    cudaStream_t stream) 
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    normalize_age_gender_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ [CUDA] Age/Gender kernel error: %s\n", cudaGetErrorString(err));
    }
}