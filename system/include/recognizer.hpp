// ============= include/recognizer.hpp =============
/*
 * ArcFace Face Recognition - TensorRT Implementation
 * 
 * CARACTERÍSTICAS:
 * - Extrae embeddings de 512 dimensiones de rostros
 * - Preprocessing GPU (resize + normalize)
 * - Inferencia TensorRT FP16
 * - Comparación por cosine similarity
 * 
 * INPUT:
 * - Rostro recortado (cualquier tamaño)
 * - Se redimensiona automáticamente a 112x112
 * 
 * OUTPUT:
 * - Vector de 512 floats (embedding)
 * - Rango normalizado: [-1, 1]
 * 
 * PERFORMANCE (Jetson Orin):
 * - ~2-3ms por rostro (GPU preprocessing)
 * - ~1-2ms por rostro (CPU preprocessing)
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <string>
#include <memory>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>


#include "tensorrt_logger.hpp"

class FaceRecognizer {
private:
    panto::TensorRTLogger logger;  // en vez de Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // GPU buffers
    void* d_input;      // Input tensor [1, 3, 112, 112]
    void* d_output;     // Output tensor [1, 512]
    void* d_resized;    // Resized image buffer
    
    cudaStream_t stream;
    cv::cuda::Stream cv_stream;
    
    // GPU preprocessing
    cv::cuda::GpuMat gpu_input;
    cv::cuda::GpuMat gpu_resized;
    
    int input_width = 112;
    int input_height = 112;
    int embedding_size = 512;
    
    bool use_gpu_preprocessing = true;
    
    // Helper functions
    bool loadEngine(const std::string& engine_path);
    cv::Mat preprocess_cpu(const cv::Mat& face);
    void preprocess_gpu(const cv::Mat& face);
    void normalize_embedding(std::vector<float>& embedding);

public:
    FaceRecognizer(const std::string& engine_path, bool gpu_preproc = true);
    ~FaceRecognizer();
    
    // Extraer embedding de un rostro
    std::vector<float> extract_embedding(const cv::Mat& face);
    
    // Comparar dos embeddings (cosine similarity)
    static float compare(const std::vector<float>& emb1, 
                        const std::vector<float>& emb2);
    
    // L2 normalize embedding
    static void l2_normalize(std::vector<float>& embedding);
    
    // Getters
    int get_embedding_size() const { return embedding_size; }
    
    // Profiling
    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};