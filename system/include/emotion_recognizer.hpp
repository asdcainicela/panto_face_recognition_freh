// ============= include/emotion_recognizer.hpp =============
/*
 * FER+ Emotion Recognition - TensorRT Implementation
 * 
 * CARACTERÍSTICAS:
 * - Clasifica rostros en 8 emociones (FER+ dataset)
 * - Input: 64x64 grayscale
 * - Output: 8 clases [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt]
 * - Preprocessing GPU (resize + grayscale + normalize)
 * 
 * MODELO: emotion-ferplus-8.onnx
 * - Input: [1, 1, 64, 64] - grayscale
 * - Normalización: [0, 1] (pixel / 255.0)
 * - Output: [1, 8] - logits
 * 
 * EMOCIONES (índices):
 * 0: neutral
 * 1: happiness
 * 2: surprise
 * 3: sadness
 * 4: anger
 * 5: disgust
 * 6: fear
 * 7: contempt
 * 
 * PERFORMANCE (Jetson Orin):
 * - ~1-2ms por rostro (GPU preprocessing)
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

enum class Emotion {
    NEUTRAL = 0,
    HAPPINESS = 1,
    SURPRISE = 2,
    SADNESS = 3,
    ANGER = 4,
    DISGUST = 5,
    FEAR = 6,
    CONTEMPT = 7
};

struct EmotionResult {
    Emotion emotion;
    float confidence;
    std::vector<float> probabilities;  // Para todas las emociones
    
    std::string to_string() const;
};

class EmotionRecognizer {
private:
    panto::TensorRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // GPU buffers
    void* d_input;      // Input tensor [1, 1, 64, 64]
    void* d_output;     // Output tensor [1, 8]
    void* d_resized;    // Resized image buffer
    
    cudaStream_t stream;
    cv::cuda::Stream cv_stream;
    
    // GPU preprocessing
    cv::cuda::GpuMat gpu_input;
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::GpuMat gpu_resized;
    
    int input_width = 64;
    int input_height = 64;
    int num_classes = 8;
    
    bool use_gpu_preprocessing = true;
    
    // Helper functions
    bool loadEngine(const std::string& engine_path);
    cv::Mat preprocess_cpu(const cv::Mat& face);
    void preprocess_gpu(const cv::Mat& face);
    EmotionResult postprocess(const std::vector<float>& logits);

public:
    EmotionRecognizer(const std::string& engine_path, bool gpu_preproc = true);
    ~EmotionRecognizer();
    
    // Predecir emoción de un rostro
    EmotionResult predict(const cv::Mat& face);
    
    // Batch prediction (más eficiente para múltiples rostros)
    std::vector<EmotionResult> predict_batch(const std::vector<cv::Mat>& faces);
    
    // Profiling
    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};

// Helper: convertir enum a string
std::string emotion_to_string(Emotion emotion);