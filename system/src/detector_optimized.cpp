// ============= src/detector_optimized.cpp =============
#include "detector_optimized.hpp"
#include "cuda_kernels.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>

constexpr int MIN_FACE_SIZE = 20;
constexpr float MAX_FACE_RATIO = 0.9f;
constexpr float MIN_ASPECT_RATIO = 0.4f;
constexpr float MAX_ASPECT_RATIO = 2.5f;
constexpr float NMS_IOM_THRESHOLD = 0.8f;

// ==================== SCRFD HELPERS ====================

static std::vector<std::vector<float>> generate_anchors_scrfd(
    int feat_h, int feat_w, int stride, int num_anchors) 
{
    std::vector<std::vector<float>> anchors;
    for (int i = 0; i < feat_h; i++) {
        for (int j = 0; j < feat_w; j++) {
            for (int k = 0; k < num_anchors; k++) {
                float cx = (j + 0.5f) * stride;
                float cy = (i + 0.5f) * stride;
                anchors.push_back({cx, cy, static_cast<float>(stride)});
            }
        }
    }
    return anchors;
}

static cv::Rect distance2bbox(const std::vector<float>& anchor, 
                              const float* distance, 
                              float scale_x, float scale_y) 
{
    float cx = anchor[0];  // Centro X del anchor
    float cy = anchor[1];  // Centro Y del anchor
    float stride = anchor[2];  // Stride del feature map
    
    // Distancias predichas (escaladas por stride)
    float l = distance[0] * stride;  // left
    float t = distance[1] * stride;  // top
    float r = distance[2] * stride;  // right
    float b = distance[3] * stride;  // bottom
    
    // Calcular bbox en coordenadas del input (640x640)
    float x1 = cx - l;
    float y1 = cy - t;
    float x2 = cx + r;
    float y2 = cy + b;
    
    // Escalar a imagen original
    x1 *= scale_x;
    y1 *= scale_y;
    x2 *= scale_x;
    y2 *= scale_y;
    
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

static void distance2kps(const std::vector<float>& anchor, 
                        const float* kps_distance, 
                        cv::Point2f* landmarks, 
                        float scale_x, float scale_y) 
{
    float cx = anchor[0];
    float cy = anchor[1];
    float stride = anchor[2];
    
    // 5 keypoints, cada uno con dx y dy
    for (int i = 0; i < 5; i++) {
        float dx = kps_distance[i * 2] * stride;
        float dy = kps_distance[i * 2 + 1] * stride;
        
        // Posición en input (640x640)
        float kx = cx + dx;
        float ky = cy + dy;
        
        // Escalar a imagen original
        landmarks[i].x = kx * scale_x;
        landmarks[i].y = ky * scale_y;
    }
}

// ==================== CONSTRUCTOR/DESTRUCTOR ====================

FaceDetectorOptimized::FaceDetectorOptimized(const std::string& engine_path, 
                                             bool gpu_preproc)
    : use_gpu_preprocessing(gpu_preproc), d_input_buffer(nullptr), 
      d_resized_buffer(nullptr)
{
    spdlog::info("🚀 [OPT] Inicializando SCRFD TensorRT Optimizado");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    // Crear stream CUDA nativo
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    
    // Crear cv::cuda::Stream wrapper para OpenCV
    cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    
    if (use_gpu_preprocessing) {
        // Pre-allocar buffers GPU
        size_t input_size = input_width * input_height * 3 * sizeof(float);
        cudaMalloc(&d_input_buffer, input_size);
        
        // Buffer para imagen resized (antes de normalizar)
        size_t resized_size = input_width * input_height * 3;
        cudaMalloc(&d_resized_buffer, resized_size);
        
        spdlog::info("   GPU buffers allocated: input={:.1f}MB, resized={:.1f}MB",
                    input_size / 1024.0 / 1024.0,
                    resized_size / 1024.0 / 1024.0);
    }
    
    spdlog::info("✓ [OPT] Detector optimizado listo");
}

FaceDetectorOptimized::~FaceDetectorOptimized() {
    for (int i = 0; i < 10; i++) {
        if (buffers[i]) cudaFree(buffers[i]);
    }
    if (d_input_buffer) cudaFree(d_input_buffer);
    if (d_resized_buffer) cudaFree(d_resized_buffer);
    if (stream) cudaStreamDestroy(stream);
}

// ==================== LOAD ENGINE ====================

bool FaceDetectorOptimized::loadEngine(const std::string& engine_path) {
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
    if (!runtime) return false;
    
    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
    if (!engine) return false;
    
    context.reset(engine->createExecutionContext());
    if (!context) return false;
    
    int nb_bindings = engine->getNbIOTensors();
    
    for (int i = 0; i < nb_bindings; i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dims = engine->getTensorShape(name);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_index = i;
            input_height = dims.d[2];
            input_width = dims.d[3];
        } else {
            output_indices.push_back(i);
        }
        
        size_t binding_size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            binding_size *= dims.d[j];
        }
        binding_size *= sizeof(float);
        
        cudaMalloc(&buffers[i], binding_size);
        context->setTensorAddress(name, buffers[i]);
    }
    
    spdlog::info("   Engine: {}x{}, {} outputs", 
                input_width, input_height, output_indices.size());
    
    return true;
}

// ==================== PREPROCESSING ====================

cv::Mat FaceDetectorOptimized::preprocess_cpu(const cv::Mat& img) {
    cv::Mat resized, normalized;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    
    // SCRFD InsightFace usa: (pixel - 127.5) / 128.0
    // PERO necesita BGR->RGB primero
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(normalized, CV_32F);
    normalized = (normalized - 127.5f) / 128.0f;
    
    return normalized;
}

void FaceDetectorOptimized::preprocess_gpu(const cv::Mat& img) {
    // Upload to GPU usando cv::cuda::Stream
    gpu_input.upload(img, cv_stream);
    
    // Resize en GPU (usa cv::cuda::Stream)
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height), 
                     0, 0, cv::INTER_LINEAR, cv_stream);
    
    // Sincronizar para obtener puntero raw
    cv_stream.waitForCompletion();
    
    // Copiar datos a buffer temporal
    cudaMemcpyAsync(d_resized_buffer, gpu_resized.data, 
                   input_width * input_height * 3,
                   cudaMemcpyDeviceToDevice, stream);
    
    // Normalización custom en GPU usando cudaStream_t nativo
    // NOTA: El kernel CUDA ya hace BGR->RGB + normalización (127.5/128.0)
    cuda_normalize_imagenet(
        static_cast<const unsigned char*>(d_resized_buffer), 
        static_cast<float*>(buffers[input_index]),
        input_width, input_height, stream
    );
}

// ==================== DETECTION ====================

std::vector<Detection> FaceDetectorOptimized::detect(const cv::Mat& img) {
    cv::Size orig_size = img.size();
    
    // === PREPROCESSING ===
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (use_gpu_preprocessing) {
        preprocess_gpu(img);
    } else {
        cv::Mat input_blob = preprocess_cpu(img);
        
        // HWC -> CHW
        std::vector<cv::Mat> channels(3);
        cv::split(input_blob, channels);
        
        size_t single_channel_size = input_height * input_width;
        std::vector<float> input_data(3 * single_channel_size);
        
        for (int c = 0; c < 3; c++) {
            std::memcpy(input_data.data() + c * single_channel_size, 
                       channels[c].data, single_channel_size * sizeof(float));
        }
        
        cudaMemcpyAsync(buffers[input_index], input_data.data(), 
                       input_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = 
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // === INFERENCE ===
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = 
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // === POSTPROCESSING ===
    std::vector<std::vector<float>> outputs(output_indices.size());
    
    for (size_t i = 0; i < output_indices.size(); i++) {
        int idx = output_indices[i];
        const char* name = engine->getIOTensorName(idx);
        auto dims = engine->getTensorShape(name);
        
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) size *= dims.d[j];
        
        outputs[i].resize(size);
        cudaMemcpyAsync(outputs[i].data(), buffers[idx],
                       size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    auto result = postprocess_scrfd(outputs, orig_size);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    last_profile.postprocess_ms = 
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    last_profile.total_ms = last_profile.preprocess_ms + 
                           last_profile.inference_ms + 
                           last_profile.postprocess_ms;
    
    return result;
}

// ==================== POSTPROCESSING ====================

std::vector<Detection> FaceDetectorOptimized::postprocess_scrfd(
    const std::vector<std::vector<float>>& outputs,
    const cv::Size& orig_size) 
{
    std::vector<Detection> detections;
    
    if (outputs.size() != 9) {
        spdlog::error("SCRFD outputs incorrectos: {} (esperados: 9)", outputs.size());
        return detections;
    }
    
    float scale_x = static_cast<float>(orig_size.width) / input_width;
    float scale_y = static_cast<float>(orig_size.height) / input_height;
    
    // Configuración de cada stride (según SCRFD)
    struct StrideConfig {
        int feat_h, feat_w, stride;
        int score_idx, bbox_idx, kps_idx;
    };
    
    std::vector<StrideConfig> strides = {
        {80, 80, 8,  0, 3, 6},   // Stride 8:  outputs[0,3,6]
        {40, 40, 16, 1, 4, 7},   // Stride 16: outputs[1,4,7]
        {20, 20, 32, 2, 5, 8}    // Stride 32: outputs[2,5,8]
    };
    
    // Procesar cada stride
    for (const auto& cfg : strides) {
        const float* scores_data = outputs[cfg.score_idx].data();
        const float* bbox_data = outputs[cfg.bbox_idx].data();
        const float* kps_data = outputs[cfg.kps_idx].data();
        
        // Generar anchor centers para este stride
        auto anchors = generate_anchors_scrfd(cfg.feat_h, cfg.feat_w, 
                                             cfg.stride, num_anchors);
        
        int total_anchors = static_cast<int>(anchors.size());
        
        // CRÍTICO: Los outputs vienen en formato [N, C] no [1, C, H, W]
        // Cada posición espacial tiene num_anchors (2) detecciones
        for (int i = 0; i < total_anchors; i++) {
            float score = scores_data[i];
            
            // Filtro por confianza
            if (score < conf_threshold) continue;
            
            // Decodificar bbox (4 valores: left, top, right, bottom distances)
            const float* bbox = &bbox_data[i * 4];
            cv::Rect box = distance2bbox(anchors[i], bbox, scale_x, scale_y);
            
            // Validaciones básicas
            if (box.width < MIN_FACE_SIZE || box.height < MIN_FACE_SIZE) continue;
            if (box.width > orig_size.width * MAX_FACE_RATIO || 
                box.height > orig_size.height * MAX_FACE_RATIO) continue;
            
            // Clipping a los límites de la imagen
            box.x = std::max(0, std::min(box.x, orig_size.width - 1));
            box.y = std::max(0, std::min(box.y, orig_size.height - 1));
            box.width = std::min(box.width, orig_size.width - box.x);
            box.height = std::min(box.height, orig_size.height - box.y);
            
            // Validar aspect ratio
            float aspect = static_cast<float>(box.width) / box.height;
            if (aspect < MIN_ASPECT_RATIO || aspect > MAX_ASPECT_RATIO) continue;
            
            // Decodificar keypoints (10 valores: 5 puntos x 2 coordenadas)
            Detection det;
            det.confidence = score;
            det.box = box;
            
            const float* kps = &kps_data[i * 10];
            distance2kps(anchors[i], kps, det.landmarks, scale_x, scale_y);
            
            detections.push_back(det);
        }
    }
    
    // Aplicar NMS
    if (!detections.empty()) {
        nms(detections);
    }
    
    return detections;
}

void FaceDetectorOptimized::nms(std::vector<Detection>& detections) {
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