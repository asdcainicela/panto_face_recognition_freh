// ============= include/age_gender_predictor.hpp =============
/*
 * Age-Gender Prediction - TensorRT Implementation
 * 
 * CARACTERÍSTICAS:
 * - Predice edad (0-100 años) y género (Male/Female)
 * - Input: 224x224 RGB
 * - Basado en modelo Hugging Face age-gender-prediction
 * - Preprocessing GPU (resize + normalize ImageNet)
 * 
 * MODELO: model_fp16.onnx
 * - Input: [1, 3, 224, 224] - RGB normalizado ImageNet
 * - Normalización: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
 * - Output: [1, num_classes] - logits
 * 
 * OUTPUT:
 * - Primeras 2 clases: género [Female, Male]
 * - Siguientes ~100 clases: edad [0-100 años]
 * 
 * PERFORMANCE (Jetson Orin):
 * - ~3-5ms por rostro (GPU preprocessing)
 * - Batch processing disponible para múltiples rostros
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

enum class Gender {
    FEMALE = 0,
    MALE = 1
};

struct AgeGenderResult {
    int age;              // Edad predicha (0-100)
    Gender gender;        // Género predicho
    float age_confidence; // Confianza en la edad
    float gender_confidence; // Confianza en el género
    
    std::string to_string() const;
};

class AgeGenderPredictor {
private:
    panto::TensorRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // GPU buffers
    void* d_input;      // Input tensor [1, 3, 224, 224]
    void* d_output;     // Output tensor [1, num_classes]
    void* d_resized;    // Resized image buffer
    
    cudaStream_t stream;
    cv::cuda::Stream cv_stream;
    
    // GPU preprocessing
    cv::cuda::GpuMat gpu_input;
    cv::cuda::GpuMat gpu_resized;
    
    int input_width = 224;
    int input_height = 224;
    int num_classes = 0;  // Se obtiene del engine
    
    bool use_gpu_preprocessing = true;
    
    // Helper functions
    bool loadEngine(const std::string& engine_path);
    cv::Mat preprocess_cpu(const cv::Mat& face);
    void preprocess_gpu(const cv::Mat& face);
    AgeGenderResult postprocess(const std::vector<float>& logits);

public:
    AgeGenderPredictor(const std::string& engine_path, bool gpu_preproc = true);
    ~AgeGenderPredictor();
    
    // Predecir edad y género de un rostro
    AgeGenderResult predict(const cv::Mat& face);
    
    // Batch prediction (más eficiente para múltiples rostros)
    std::vector<AgeGenderResult> predict_batch(const std::vector<cv::Mat>& faces);
    
    // Profiling
    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};

// Helper: convertir enum a string
std::string gender_to_string(Gender gender);