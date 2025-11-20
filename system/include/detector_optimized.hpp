// ============= include/detector_optimized.hpp =============
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "tensorrt_logger.hpp"


struct Detection {
    cv::Rect box;
    float confidence;
    cv::Point2f landmarks[5];
};


class FaceDetectorOptimized {
private:
    panto::TensorRTLogger logger;  // en vez de Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // GPU buffers
    void* buffers[10];
    void* d_input_buffer;      // Input tensor en GPU
    void* d_resized_buffer;    // Imagen resized temporal
    cudaStream_t stream;       // CUDA stream nativo (para TensorRT)
    cv::cuda::Stream cv_stream; // OpenCV CUDA stream wrapper
    
    // GPU preprocessing buffers
    cv::cuda::GpuMat gpu_input;
    cv::cuda::GpuMat gpu_resized;
    
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.4f;
    
    int input_index = -1;
    std::vector<int> output_indices;
    
    std::vector<int> feat_stride_fpn = {8, 16, 32};
    int num_anchors = 2;
    
    bool use_gpu_preprocessing = true;
    
    cv::Mat preprocess_cpu(const cv::Mat& img);
    void preprocess_gpu(const cv::Mat& img);
    std::vector<Detection> postprocess_scrfd(
        const std::vector<std::vector<float>>& outputs,
        const cv::Size& orig_size);
    void nms(std::vector<Detection>& detections);
    bool loadEngine(const std::string& engine_path);

public:
    FaceDetectorOptimized(const std::string& engine_path, 
                         bool gpu_preproc = true);
    ~FaceDetectorOptimized();
    
    std::vector<Detection> detect(const cv::Mat& img);
    
    void set_conf_threshold(float threshold) { conf_threshold = threshold; }
    void set_nms_threshold(float threshold) { nms_threshold = threshold; }
    void enable_gpu_preprocessing(bool enable) { use_gpu_preprocessing = enable; }
    
    // Profiling
    struct ProfileStats {
        double preprocess_ms = 0;
        double inference_ms = 0;
        double postprocess_ms = 0;
        double total_ms = 0;
    };
    ProfileStats last_profile;
};