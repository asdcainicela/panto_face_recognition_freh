#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

FaceDetector::FaceDetector(const std::string& model_path, bool use_cuda) 
    : env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector") {
    
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            spdlog::info("detector: cuda habilitado");
        } catch (const std::exception& e) {
            spdlog::warn("detector: cuda falló, usando cpu: {}", e.what());
        }
    }
    
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        spdlog::info("detector: modelo cargado desde {}", model_path);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names
        size_t num_input = session->GetInputCount();
        for (size_t i = 0; i < num_input; i++) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(strdup(name.get()));
        }
        
        // Output names
        size_t num_output = session->GetOutputCount();
        for (size_t i = 0; i < num_output; i++) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(strdup(name.get()));
        }
        
        spdlog::info("detector: {} inputs, {} outputs", num_input, num_output);
        
        // Detectar input size del modelo
        auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() == 4) {
            input_height = static_cast<int>(input_shape[2]);
            input_width = static_cast<int>(input_shape[3]);
            spdlog::info("detector: input size {}x{}", input_width, input_height);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("detector: error al cargar modelo: {}", e.what());
        throw;
    }
}

FaceDetector::~FaceDetector() {
    for (auto name : input_names) free(const_cast<char*>(name));
    for (auto name : output_names) free(const_cast<char*>(name));
}

cv::Mat FaceDetector::preprocess(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    
    // Normalizar [0, 255] -> [-1, 1] (det_10g usa mean/std normalization)
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F);
    
    // BGR -> RGB
    cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
    
    // Normalización: (pixel - 127.5) / 128.0
    normalized = (normalized - 127.5) / 128.0;
    
    return normalized;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& img) {
    cv::Size orig_size = img.size();
    cv::Mat input_blob = preprocess(img);
    
    // Crear tensor [1, 3, H, W]
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    size_t input_size = 1 * 3 * input_height * input_width;
    
    std::vector<float> input_data(input_size);
    
    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(input_blob, channels);
    
    size_t single_channel_size = input_height * input_width;
    for (int c = 0; c < 3; c++) {
        std::memcpy(input_data.data() + c * single_channel_size, 
                   channels[c].data, 
                   single_channel_size * sizeof(float));
    }
    
    // Crear tensor ONNX
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size, 
        input_shape.data(), input_shape.size()
    );
    
    // Inferencia
    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, input_names.size(),
            output_names.data(), output_names.size()
        );
        
        return postprocess(outputs, orig_size);
        
    } catch (const std::exception& e) {
        spdlog::error("detector: inferencia falló: {}", e.what());
        return {};
    }
}

std::vector<Detection> FaceDetector::postprocess(const std::vector<Ort::Value>& outputs,
                                                 const cv::Size& orig_size) {
    std::vector<Detection> detections;
    
    // det_10g.onnx produce:
    // Output 0: scores [N, 1] o [N, 2]
    // Output 1: boxes [N, 4] 
    // Output 2: kpss [N, 10] (5 landmarks * 2 coords)
    
    if (outputs.size() < 3) {
        spdlog::warn("detector: outputs insuficientes: {}", outputs.size());
        return detections;
    }
    
    // Extraer shapes
    auto scores_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto boxes_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    auto kpss_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    
    int num_detections = static_cast<int>(scores_shape[0]);
    
    if (num_detections == 0) {
        return detections;
    }
    
    auto* scores_data = outputs[0].GetTensorData<float>();
    auto* boxes_data = outputs[1].GetTensorData<float>();
    auto* kpss_data = outputs[2].GetTensorData<float>();
    
    spdlog::debug("detector: {} candidatos iniciales", num_detections);
    
    float scale_x = static_cast<float>(orig_size.width) / input_width;
    float scale_y = static_cast<float>(orig_size.height) / input_height;
    
    // Procesar detecciones
    for (int i = 0; i < num_detections; i++) {
        float score = scores_data[i];
        
        if (score < conf_threshold) continue;
        
        Detection det;
        det.confidence = score;
        
        // Boxes: [x1, y1, x2, y2]
        float x1 = boxes_data[i * 4 + 0] * scale_x;
        float y1 = boxes_data[i * 4 + 1] * scale_y;
        float x2 = boxes_data[i * 4 + 2] * scale_x;
        float y2 = boxes_data[i * 4 + 3] * scale_y;
        
        // Validar coordenadas
        x1 = std::max(0.0f, std::min(x1, (float)orig_size.width));
        y1 = std::max(0.0f, std::min(y1, (float)orig_size.height));
        x2 = std::max(0.0f, std::min(x2, (float)orig_size.width));
        y2 = std::max(0.0f, std::min(y2, (float)orig_size.height));
        
        if (x2 <= x1 || y2 <= y1) continue;
        
        float width = x2 - x1;
        float height = y2 - y1;
        if (width < 10 || height < 10) continue;
        
        det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        
        // Landmarks: [x0, y0, x1, y1, ..., x4, y4]
        for (int j = 0; j < 5; j++) {
            det.landmarks[j].x = kpss_data[i * 10 + j * 2] * scale_x;
            det.landmarks[j].y = kpss_data[i * 10 + j * 2 + 1] * scale_y;
        }
        
        detections.push_back(det);
    }
    
    spdlog::debug("detector: {} detecciones antes de NMS", detections.size());
    nms(detections);
    spdlog::info("detector: {} rostros detectados", detections.size());
    
    return detections;
}

void FaceDetector::nms(std::vector<Detection>& detections) {
    std::sort(detections.begin(), detections.end(), 
             [](const Detection& a, const Detection& b) {
                 return a.confidence > b.confidence;
             });
    
    std::vector<Detection> kept;
    
    for (size_t i = 0; i < detections.size(); i++) {
        bool keep = true;
        
        for (size_t j = 0; j < kept.size(); j++) {
            float intersection = static_cast<float>((detections[i].box & kept[j].box).area());
            float union_area = static_cast<float>(detections[i].box.area() + kept[j].box.area() - intersection);
            
            if (union_area == 0) continue;
            
            float iou = intersection / union_area;
            
            if (iou > nms_threshold) {
                keep = false;
                break;
            }
        }
        
        if (keep) kept.push_back(detections[i]);
    }
    
    detections = kept;
}