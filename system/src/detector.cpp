#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

constexpr int MIN_FACE_SIZE = 20;
constexpr float MAX_FACE_RATIO = 0.9f;
constexpr float MIN_ASPECT_RATIO = 0.4f;
constexpr float MAX_ASPECT_RATIO = 2.5f;
constexpr int MIN_LANDMARKS_INSIDE = 2;
constexpr bool ENABLE_LANDMARK_VALIDATION = false;
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

FaceDetector::FaceDetector(const std::string& model_path, bool use_cuda) 
    : env(ORT_LOGGING_LEVEL_WARNING, "FaceDetector") {
    
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    bool cuda_available = false;
    
    if (use_cuda) {
        try {
            // Verificar si CUDA esta disponible
            auto available_providers = Ort::GetAvailableProviders();
            bool has_cuda = false;
            
            spdlog::info("Providers disponibles:");
            for (const auto& provider : available_providers) {
                spdlog::info("  - {}", provider);
                if (provider == "CUDAExecutionProvider") {
                    has_cuda = true;
                }
            }
            
            if (has_cuda) {
                // Metodo simple: configuracion minima
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                
                session_options.AppendExecutionProvider_CUDA(cuda_opts);
                cuda_available = true;
                spdlog::info("CUDA Provider habilitado");
            } else {
                spdlog::warn("CUDAExecutionProvider no encontrado");
            }
            
        } catch (const Ort::Exception& e) {
            spdlog::warn("No se pudo habilitar CUDA: {} (codigo: {})", e.what(), e.GetOrtErrorCode());
            spdlog::info("Continuando con CPU...");
        } catch (const std::exception& e) {
            spdlog::warn("Error configurando CUDA: {}", e.what());
        }
    }
    
    // Fallback: intentar con configuracion minimalista
    if (use_cuda && !cuda_available) {
        try {
            OrtCUDAProviderOptions minimal_opts{};
            minimal_opts.device_id = 0;
            minimal_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            minimal_opts.do_copy_in_default_stream = 1;
            
            session_options.AppendExecutionProvider_CUDA(minimal_opts);
            cuda_available = true;
            spdlog::info("CUDA habilitado (configuracion minima)");
            
        } catch (...) {
            spdlog::info("CUDA no disponible, usando CPU");
        }
    }
    
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        if (cuda_available) {
            spdlog::info("Detector inicializado con aceleracion CUDA");
        } else {
            spdlog::warn("DETECTOR USANDO CPU (muy lento)");
            spdlog::warn("Para habilitar CUDA:");
            spdlog::warn("  1. Verificar: python3 -c \"import onnxruntime; print(onnxruntime.get_available_providers())\"");
            spdlog::warn("  2. Debe mostrar: ['CUDAExecutionProvider', 'CPUExecutionProvider']");
        }
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_input = session->GetInputCount();
        for (size_t i = 0; i < num_input; i++) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(strdup(name.get()));
        }
        
        size_t num_output = session->GetOutputCount();
        for (size_t i = 0; i < num_output; i++) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(strdup(name.get()));
        }
        
        // Verificar que tengamos 9 outputs
        if (num_output != 9) {
            spdlog::warn("Modelo retorna {} outputs (esperados: 9)", num_output);
            spdlog::warn("Esto puede indicar un modelo incorrecto");
            spdlog::warn("Descarga det_10g.onnx desde buffalo_l");
        }
        
        input_width = 640;
        input_height = 640;
        conf_threshold = 0.5f;
        nms_threshold = 0.3f;
        
    } catch (const Ort::Exception& e) {
        spdlog::error("Error cargando modelo: {} (codigo: {})", e.what(), e.GetOrtErrorCode());
        throw;
    } catch (const std::exception& e) {
        spdlog::error("Error cargando modelo: {}", e.what());
        throw;
    }
}

FaceDetector::~FaceDetector() {
    for (auto name : input_names) free(const_cast<char*>(name));
    for (auto name : output_names) free(const_cast<char*>(name));
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
    
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
    size_t input_size = 1 * 3 * input_height * input_width;
    
    std::vector<float> input_data(input_size);
    std::vector<cv::Mat> channels(3);
    cv::split(input_blob, channels);
    
    size_t single_channel_size = input_height * input_width;
    for (int c = 0; c < 3; c++) {
        std::memcpy(input_data.data() + c * single_channel_size, 
                   channels[c].data, 
                   single_channel_size * sizeof(float));
    }
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size, 
        input_shape.data(), input_shape.size()
    );
    
    try {
        auto outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, input_names.size(),
            output_names.data(), output_names.size()
        );
        
        return postprocess(outputs, orig_size);
        
    } catch (const std::exception& e) {
        spdlog::error("Inferencia fallo: {}", e.what());
        return {};
    }
}

std::vector<Detection> FaceDetector::postprocess(const std::vector<Ort::Value>& outputs,
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
        
        auto* scores_data = outputs[score_idx].GetTensorData<float>();
        auto* boxes_data = outputs[box_idx].GetTensorData<float>();
        auto* kpss_data = outputs[kpss_idx].GetTensorData<float>();
        
        auto scores_shape = outputs[score_idx].GetTensorTypeAndShapeInfo().GetShape();
        int num_anchors = static_cast<int>(scores_shape[0]);
        
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
            
            if (ENABLE_LANDMARK_VALIDATION) {
                int landmarks_inside = 0;
                for (int j = 0; j < 5; j++) {
                    if (det.landmarks[j].x >= det.box.x && 
                        det.landmarks[j].x <= det.box.x + det.box.width &&
                        det.landmarks[j].y >= det.box.y && 
                        det.landmarks[j].y <= det.box.y + det.box.height) {
                        landmarks_inside++;
                    }
                }
                if (landmarks_inside < MIN_LANDMARKS_INSIDE) continue;
            }
            
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