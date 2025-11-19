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