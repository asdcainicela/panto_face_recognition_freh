#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

// Generar anchors para una escala específica
static std::vector<std::vector<float>> generate_anchors(int feat_h, int feat_w, int stride) {
    std::vector<std::vector<float>> anchors;
    
    // det_10g usa 2 anchors por celda (ratio 1:1)
    std::vector<int> anchor_sizes = {stride * 1, stride * 2};
    
    for (int i = 0; i < feat_h; i++) {
        for (int j = 0; j < feat_w; j++) {
            // Centro del anchor
            float cx = (j + 0.5f) * stride;
            float cy = (i + 0.5f) * stride;
            
            // 2 anchors por celda con diferentes tamaños
            for (int anchor_size : anchor_sizes) {
                anchors.push_back({cx, cy, static_cast<float>(anchor_size), static_cast<float>(anchor_size)});
            }
        }
    }
    
    return anchors;
}

// Decodificar box desde offset + anchor
static cv::Rect decode_box(const std::vector<float>& anchor, const float* bbox_pred, float scale_x, float scale_y) {
    float cx = anchor[0];
    float cy = anchor[1];
    float w = anchor[2];
    float h = anchor[3];
    
    // Decodificar offsets (formato RetinaFace)
    float pred_cx = cx + bbox_pred[0] * w;
    float pred_cy = cy + bbox_pred[1] * h;
    float pred_w = std::exp(bbox_pred[2]) * w;
    float pred_h = std::exp(bbox_pred[3]) * h;
    
    // Convertir de centro-ancho a x1y1x2y2
    float x1 = (pred_cx - pred_w * 0.5f) * scale_x;
    float y1 = (pred_cy - pred_h * 0.5f) * scale_y;
    float x2 = (pred_cx + pred_w * 0.5f) * scale_x;
    float y2 = (pred_cy + pred_h * 0.5f) * scale_y;
    
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

// Decodificar landmarks desde offset + anchor
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
        
        // det_10g tiene input dinámico, usamos 640x640
        input_width = 640;
        input_height = 640;
        
        // Threshold más alto para reducir falsos positivos
        conf_threshold = 0.5f;
        nms_threshold = 0.3f;
        
        spdlog::info("detector: usando input size {}x{}, conf={:.2f}, nms={:.2f}", 
                    input_width, input_height, conf_threshold, nms_threshold);
        
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
    
    // Normalizar [0, 255] -> [-1, 1]
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32F);
    
    // BGR -> RGB
    cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
    
    // (pixel - 127.5) / 128.0
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
    
    if (outputs.size() != 9) {
        spdlog::error("detector: se esperaban 9 outputs, recibidos: {}", outputs.size());
        return detections;
    }
    
    float scale_x = static_cast<float>(orig_size.width) / input_width;
    float scale_y = static_cast<float>(orig_size.height) / input_height;
    
    // Strides y tamaños de feature maps para det_10g
    // 640x640 input → [80x80, 40x40, 20x20] feature maps
    struct ScaleInfo {
        int feat_h, feat_w, stride;
    };
    
    std::vector<ScaleInfo> scales = {
        {80, 80, 8},   // 640/8 = 80
        {40, 40, 16},  // 640/16 = 40
        {20, 20, 32}   // 640/32 = 20
    };
    
    // Procesar las 3 escalas
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int score_idx = scale_idx;           // 0, 1, 2
        int box_idx = scale_idx + 3;         // 3, 4, 5
        int kpss_idx = scale_idx + 6;        // 6, 7, 8
        
        auto* scores_data = outputs[score_idx].GetTensorData<float>();
        auto* boxes_data = outputs[box_idx].GetTensorData<float>();
        auto* kpss_data = outputs[kpss_idx].GetTensorData<float>();
        
        auto scores_shape = outputs[score_idx].GetTensorTypeAndShapeInfo().GetShape();
        int num_anchors = static_cast<int>(scores_shape[0]);
        
        // Generar anchors para esta escala
        auto anchors = generate_anchors(scales[scale_idx].feat_h, 
                                       scales[scale_idx].feat_w, 
                                       scales[scale_idx].stride);
        
        if (anchors.size() != static_cast<size_t>(num_anchors)) {
            spdlog::warn("detector: mismatch anchors escala {}: generados {} vs esperados {}", 
                        scale_idx, anchors.size(), num_anchors);
            continue;
        }
        
        // Procesar cada anchor
        for (int i = 0; i < num_anchors; i++) {
            float score = scores_data[i];
            
            if (score < conf_threshold) continue;
            
            Detection det;
            det.confidence = score;
            
            // Decodificar box
            det.box = decode_box(anchors[i], &boxes_data[i * 4], scale_x, scale_y);
            
            // Validar box - Más estricto
            if (det.box.width < 30 || det.box.height < 30) continue;  // Min 30px
            if (det.box.width > orig_size.width * 0.8) continue;  // Max 80% del frame
            if (det.box.height > orig_size.height * 0.8) continue;
            if (det.box.x < 0 || det.box.y < 0) continue;
            if (det.box.x + det.box.width > orig_size.width) continue;
            if (det.box.y + det.box.height > orig_size.height) continue;
            
            // Validar aspect ratio (rostros son ~1:1.2)
            float aspect = static_cast<float>(det.box.width) / det.box.height;
            if (aspect < 0.5f || aspect > 2.0f) continue;
            
            // Validar que los landmarks estén dentro del box (filtro extra de calidad)
            int landmarks_inside = 0;
            for (int j = 0; j < 5; j++) {
                if (det.landmarks[j].x >= det.box.x && 
                    det.landmarks[j].x <= det.box.x + det.box.width &&
                    det.landmarks[j].y >= det.box.y && 
                    det.landmarks[j].y <= det.box.y + det.box.height) {
                    landmarks_inside++;
                }
            }
            // Al menos 3 de 5 landmarks deben estar dentro del box
            if (landmarks_inside < 3) {
                spdlog::debug("descartado: solo {}/5 landmarks dentro del box", landmarks_inside);
                continue;
            }
            
            // Decodificar landmarks
            decode_landmarks(anchors[i], &kpss_data[i * 10], det.landmarks, scale_x, scale_y);
            
            // Debug: Log detecciones con score alto
            if (score > 0.4f) {
                spdlog::info("  Escala {} anchor {}: score={:.3f}, box=[{},{},{},{}]",
                            scale_idx, i, score, 
                            det.box.x, det.box.y, det.box.width, det.box.height);
            }
            
            detections.push_back(det);
        }
    }
    
    spdlog::debug("detector: {} detecciones antes de NMS", detections.size());
    
    if (!detections.empty()) {
        nms(detections);
        spdlog::info("detector: {} rostros detectados", detections.size());
    } else {
        spdlog::warn("detector: no se detectaron rostros (threshold={:.2f})", conf_threshold);
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
            
            // También considerar si uno está contenido dentro del otro (IoM)
            float iom_i = intersection / area_i;  // Intersection over Min
            float iom_j = intersection / area_j;
            
            // Suprimir si:
            // 1. IOU normal es alto (boxes muy solapados)
            // 2. Uno está casi completamente dentro del otro
            if (iou > nms_threshold || iom_i > 0.7f || iom_j > 0.7f) {
                suppressed[j] = true;
            }
        }
    }
    
    detections = kept;
}