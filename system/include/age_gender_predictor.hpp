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
#include "cuda_kernels.h"

enum class Gender {
    FEMALE = 0,
    MALE = 1
};

struct AgeGenderResult {
    int age;
    Gender gender;
    float age_confidence;
    float gender_confidence;
    std::string to_string() const;
};

class AgeGenderPredictor {
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
    cv::cuda::GpuMat gpu_resized;

    int input_width = 224;
    int input_height = 224;
    int num_classes = 0;

    bool use_gpu_preprocessing = true;

    bool loadEngine(const std::string& engine_path);
    cv::Mat preprocess_cpu(const cv::Mat& face);
    void preprocess_gpu(const cv::Mat& face);
    AgeGenderResult postprocess(const std::vector<float>& logits);

public:
    AgeGenderPredictor(const std::string& engine_path, bool gpu_preproc = true);
    ~AgeGenderPredictor();

    AgeGenderResult predict(const cv::Mat& face);
    std::vector<AgeGenderResult> predict_batch(const std::vector<cv::Mat>& faces);

    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};

std::string gender_to_string(Gender gender);