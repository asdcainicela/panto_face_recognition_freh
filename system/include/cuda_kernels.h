#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Normalización ImageNet en GPU (BGR -> RGB + normalize) - SCRFD
void cuda_normalize_imagenet(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height,
    cudaStream_t stream
);

// Normalización Age/Gender (ImageNet normalization)
void cuda_normalize_age_gender(
    const unsigned char* d_input,
    float* d_output,
    int width,
    int height,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif