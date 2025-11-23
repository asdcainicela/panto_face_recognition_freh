// ============= include/recognition/emotion_recognizer.hpp =============
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

#include "core/tensorrt_logger.hpp"  

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
    std::vector<float> probabilities;
    std::string to_string() const;
};

class EmotionRecognizer {
private:
    panto::TensorRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    void* d_input;
    void* d_output;
    void* d_resized;

    cudaStream_t stream;

    cv::cuda::GpuMat gpu_input;
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::GpuMat gpu_resized;

    int input_width = 64;
    int input_height = 64;
    int num_classes = 8;

    bool use_gpu_preprocessing = true;

    bool loadEngine(const std::string& engine_path);
    cv::Mat preprocess_cpu(const cv::Mat& face);
    void preprocess_gpu(const cv::Mat& face);
    EmotionResult postprocess(const std::vector<float>& logits);

public:
    EmotionRecognizer(const std::string& engine_path, bool gpu_preproc = true);
    ~EmotionRecognizer();

    EmotionResult predict(const cv::Mat& face);
    std::vector<EmotionResult> predict_batch(const std::vector<cv::Mat>& faces);

    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};

std::string emotion_to_string(Emotion emotion);
