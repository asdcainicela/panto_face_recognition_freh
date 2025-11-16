#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>

constexpr int MIN_FACE_SIZE = 20;
constexpr float MAX_FACE_RATIO = 0.9f;
constexpr float MIN_ASPECT_RATIO = 0.4f;
constexpr float MAX_ASPECT_RATIO = 2.5f;
constexpr float NMS_IOM_THRESHOLD = 0.8f;

static std::vector<std::vector<float>> generate_anchors(int feat_h, int feat_w, int stride) {
    std::vector<std::vector<float>> anchors;
    std::vector<int> anchor_sizes = {stride * 1, stride * 2};
    
    for (int i = 0; i < feat_h; i++) {
        for (int j = 0; j < feat_w; j++) {
            float cx = (j + 0.5f) * stride;
            float cy = (i + 0.5f) * stride;
            
            for (int anchor_size : anchor_sizes) {
                anchors.push_back({cx, cy, static_cast<float>(anchor_size), static_cast<float>(anchor_size)});
            }
        }
    }
    
    return anchors;
}

static cv::Rect decode_box(const std::vector<float>& anchor, const float* bbox_pred, float scale_x, float scale_y) {
    float cx = anchor[0];
    float cy = anchor[1];
    float w = anchor[2];
    float h = anchor[3];
    
    float pred_cx = cx + bbox_pred[0] * w;
    float pred_cy = cy + bbox_pred[1] * h;
    float pred_w = std::exp(bbox_pred[2]) * w;
    float pred_h = std::exp(bbox_pred[3]) * h;
    
    float x1 = (pred_cx - pred_w * 0.5f) * scale_x;
    float y1 = (pred_cy - pred_h * 0.5f) * scale_y;
    float x2 = (pred_cx + pred_w * 0.5f) * scale_x;
    float y2 = (pred_cy + pred_h * 0.5f) * scale_y;
    
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

static void decode_landmarks(const std::vector<float>& anchor, const float* kpss_pred, 
                            cv::Point2f* landmarks, float scale_x, float scale_y) {
    float cx = anchor[0];
    float cy = anchor[1];
    float w = anchor[2];
    float h = anchor[3];
    
    for (int i = 0; i < 5; i++) {
        float lm_x = (cx + kpss_pred[i * 2] * w) * scale_x;
        float lm_y = (cy + kpss_pred[i * 2 + 1] * h) * scale_y;
        landmarks[i] = cv::Point2f(lm_x, lm_y);
    }
}

FaceDetector::FaceDetector(const std::string& engine_path) {
    spdlog::info("Cargando TensorRT engine: {}", engine_path);
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    // Crear CUDA stream
    cudaStreamCreate(&stream);
    
    spdlog::info("Detector TensorRT inicializado correctamente");
}

FaceDetector::~FaceDetector() {
    // Liberar buffers GPU
    for (int i = 0; i < 10; i++) {
        if (buffers[i]) {
            cudaFree(buffers[i]);
        }
    }
    
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

bool FaceDetector::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        spdlog::error("No se puede abrir engine: {}", engine_path);
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        spdlog::error("No se pudo crear runtime");
        return false;
    }
    
    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!engine) {
        spdlog::error("No se pudo deserializar engine");
        return false;
    }
    
    context.reset(engine->createExecutionContext());
    if (!context) {
        spdlog::error("No se pudo crear execution context");
        return false;
    }
    
    // Identificar input y outputs
    int nb_bindings = engine->getNbBindings();
    spdlog::info("Engine tiene {} bindings", nb_bindings);
    
    for (int i = 0; i < nb_bindings; i++) {
        auto dims = engine->getBindingDimensions(i);
        auto dtype = engine->getBindingDataType(i);
        bool is_input = engine->bindingIsInput(i);
        
        spdlog::info("Binding {}: {} [{}] - {}", 
                    i, engine->getBindingName(i),
                    is_input ? "INPUT" : "OUTPUT",
                    dims.d[0]);
        
        if (is_input) {
            input_index = i;
            input_height = dims.d[2];
            input_width = dims.d[3];
        } else {
            output_indices.push_back(i);
        }
        
        // Alocar memoria GPU
        size_t binding_size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            binding_size *= dims.d[j];
        }
        binding_size *= sizeof(float);
        
        cudaMalloc(&buffers[i], binding_size);
    }
    
    spdlog::info("Input: {}x{}", input_width, input_height);
    spdlog::info("Outputs: {} tensors", output_indices.size());
    
    return true;
}

cv::Mat FaceDetector::preprocess(const cv::Mat& img) {
    cv::Mat resized, normalized;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    resized.convertTo(normalized, CV_32F);
    cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
    normalized = (normalized - 127.5) / 128.0;
    return normalized;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& img) {
    cv::Size orig_size = img.size();
    cv::Mat input_blob = preprocess(img);
    
    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(input_blob, channels);
    
    size_t single_channel_size = input_height * input_width;
    std::vector<float> input_data(1 * 3 * input_height * input_width);
    
    for (int c = 0; c < 3; c++) {
        std::memcpy(input_data.data() + c * single_channel_size, 
                   channels[c].data, 
                   single_channel_size * sizeof(float));
    }
    
    // Copiar input a GPU
    cudaMemcpyAsync(buffers[input_index], input_data.data(), 
                    input_data.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    // Ejecutar inferencia
    context->enqueueV2(buffers, stream, nullptr);
    
    // Copiar outputs de GPU a CPU
    std::vector<std::vector<float>> outputs(output_indices.size());
    
    for (size_t i = 0; i < output_indices.size(); i++) {
        int idx = output_indices[i];
        auto dims = engine->getBindingDimensions(idx);
        
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= dims.d[j];
        }
        
        outputs[i].resize(size);
        cudaMemcpyAsync(outputs[i].data(), buffers[idx],
                       size * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    return postprocess(outputs, orig_size);
}

std::vector<Detection> FaceDetector::postprocess(const std::vector<std::vector<float>>& outputs,
                                                 const cv::Size& orig_size) {
    std::vector<Detection> detections;
    
    if (outputs.size() != 9) {
        spdlog::error("Outputs incorrectos: {}", outputs.size());
        return detections;
    }
    
    float scale_x = static_cast<float>(orig_size.width) / input_width;
    float scale_y = static_cast<float>(orig_size.height) / input_height;
    
    struct ScaleInfo {
        int feat_h, feat_w, stride;
    };
    
    std::vector<ScaleInfo> scales = {
        {80, 80, 8},
        {40, 40, 16},
        {20, 20, 32}
    };
    
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int score_idx = scale_idx;
        int box_idx = scale_idx + 3;
        int kpss_idx = scale_idx + 6;
        
        const float* scores_data = outputs[score_idx].data();
        const float* boxes_data = outputs[box_idx].data();
        const float* kpss_data = outputs[kpss_idx].data();
        
        int num_anchors = static_cast<int>(outputs[score_idx].size());
        
        auto anchors = generate_anchors(scales[scale_idx].feat_h, 
                                       scales[scale_idx].feat_w, 
                                       scales[scale_idx].stride);
        
        if (anchors.size() != static_cast<size_t>(num_anchors)) continue;
        
        for (int i = 0; i < num_anchors; i++) {
            float score = scores_data[i];
            if (score < conf_threshold) continue;
            
            Detection det;
            det.confidence = score;
            det.box = decode_box(anchors[i], &boxes_data[i * 4], scale_x, scale_y);
            decode_landmarks(anchors[i], &kpss_data[i * 10], det.landmarks, scale_x, scale_y);
            
            if (det.box.width < MIN_FACE_SIZE || det.box.height < MIN_FACE_SIZE) continue;
            if (det.box.width > orig_size.width * MAX_FACE_RATIO || 
                det.box.height > orig_size.height * MAX_FACE_RATIO) continue;
            if (det.box.x < 0 || det.box.y < 0 || 
                det.box.x + det.box.width > orig_size.width || 
                det.box.y + det.box.height > orig_size.height) continue;
            
            float aspect = static_cast<float>(det.box.width) / det.box.height;
            if (aspect < MIN_ASPECT_RATIO || aspect > MAX_ASPECT_RATIO) continue;
            
            detections.push_back(det);
        }
    }
    
    if (!detections.empty()) {
        nms(detections);
    }
    
    return detections;
}

void FaceDetector::nms(std::vector<Detection>& detections) {
    std::sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) {
                 return a.confidence > b.confidence;
             });
    
    std::vector<Detection> kept;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            float intersection = static_cast<float>((detections[i].box & detections[j].box).area());
            float area_i = static_cast<float>(detections[i].box.area());
            float area_j = static_cast<float>(detections[j].box.area());
            float union_area = area_i + area_j - intersection;
            
            if (union_area <= 0) continue;
            
            float iou = intersection / union_area;
            float iom_i = intersection / area_i;
            float iom_j = intersection / area_j;
            
            if (iou > nms_threshold || iom_i > NMS_IOM_THRESHOLD || iom_j > NMS_IOM_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }
    
    detections = kept;
}