#include <cuda_runtime.h>
#include <cstdio>

// ==================== KERNEL SCRFD - CORREGIDO ====================
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
    
    // ✅ NO dividir por 255 primero, usar valores raw [0-255]
    float b = static_cast<float>(input[pixel_idx + 0]);
    float g = static_cast<float>(input[pixel_idx + 1]);
    float r = static_cast<float>(input[pixel_idx + 2]);
    
    // ✅ Normalizar directamente: (pixel - mean) / std
    int plane_size = width * height;
    output[0 * plane_size + idx] = (r - mean_r) / std_r;  // R channel
    output[1 * plane_size + idx] = (g - mean_g) / std_g;  // G channel
    output[2 * plane_size + idx] = (b - mean_b) / std_b;  // B channel
}

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
    
    // ✅ SCRFD normalization: (pixel - 127.5) / 128.0
    const float mean_r = 127.5f;
    const float mean_g = 127.5f;
    const float mean_b = 127.5f;
    const float std_r = 128.0f;
    const float std_g = 128.0f;
    const float std_b = 128.0f;
    
    normalize_imagenet_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        mean_r, mean_g, mean_b, std_r, std_g, std_b
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}

// ==================== KERNEL AGE/GENDER ====================
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
    
    // ✅ BGR -> RGB + normalize to [0, 1] primero
    float b = input[pixel_idx + 0] / 255.0f;
    float g = input[pixel_idx + 1] / 255.0f;
    float r = input[pixel_idx + 2] / 255.0f;
    
    // ✅ Luego aplicar ImageNet normalization
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
        printf("❌ CUDA kernel error (age/gender): %s\n", cudaGetErrorString(err));
    }
}