#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

// TensorRT headers
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

struct Detection {
    cv::Rect box;
    float confidence;
    cv::Point2f landmarks[5];
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

class FaceDetector {
private:
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // Buffers GPU
    void* buffers[10];  // input + outputs
    cudaStream_t stream;
    
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.6f;
    float nms_threshold = 0.4f;
    
    int input_index = -1;
    std::vector<int> output_indices;
    
    cv::Mat preprocess(const cv::Mat& img);
    std::vector<Detection> postprocess(const std::vector<std::vector<float>>& outputs,
                                      const cv::Size& orig_size);
    void nms(std::vector<Detection>& detections);
    
    bool loadEngine(const std::string& engine_path);

public:
    FaceDetector(const std::string& engine_path);
    ~FaceDetector();
    
    std::vector<Detection> detect(const cv::Mat& img);
    
    void set_conf_threshold(float threshold) { conf_threshold = threshold; }
    void set_nms_threshold(float threshold) { nms_threshold = threshold; }
};