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