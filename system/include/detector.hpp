#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

struct Detection {
    cv::Rect box;           // Bounding box
    float confidence;       // Confianza
    cv::Point2f landmarks[5]; // 5 puntos faciales (ojos, nariz, boca)
};

class FaceDetector {
private:
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::SessionOptions session_options;
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.6f;
    float nms_threshold = 0.4f;
    
    cv::Mat preprocess(const cv::Mat& img);
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputs, 
                                      const cv::Size& orig_size);
    void nms(std::vector<Detection>& detections);

public:
    FaceDetector(const std::string& model_path, bool use_cuda = true);
    ~FaceDetector();
    
    std::vector<Detection> detect(const cv::Mat& img);
    
    void set_conf_threshold(float threshold) { conf_threshold = threshold; }
    void set_nms_threshold(float threshold) { nms_threshold = threshold; }
};