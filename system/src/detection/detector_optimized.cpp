// ============= src/detection/detector_optimized.cpp - FIXED SYNC =============
/*
 * CORRECCIÓN CRÍTICA DE SINCRONIZACIÓN (2025-11-23)
 * 
 * PROBLEMA IDENTIFICADO:
 * - cv_stream (OpenCV) y stream (CUDA nativo) son diferentes streams
 * - cudaMemcpyAsync con stream después de operaciones en cv_stream causaba race conditions
 * - El video se pegaba porque las operaciones GPU no estaban sincronizadas
 * 
 * SOLUCIÓN:
 * ✅ Usar SOLO cv_stream para TODAS las operaciones asíncronas
 * ✅ Sincronizar UNA VEZ después de preprocess_gpu() antes de inference
 * ✅ Eliminar todas las sincronizaciones intermedias
 */

#include "detection/detector_optimized.hpp"
#include "detection/cuda_kernels.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>

constexpr int MIN_FACE_SIZE = 20;
constexpr float MAX_FACE_RATIO = 0.9f;
constexpr float MIN_ASPECT_RATIO = 0.4f;
constexpr float MAX_ASPECT_RATIO = 2.5f;
constexpr float NMS_IOM_THRESHOLD = 0.8f;

// ==================== SCRFD HELPERS (sin cambios) ====================

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
    float cx = anchor[0];
    float cy = anchor[1];
    float stride = anchor[2];
    
    float l = distance[0] * stride;
    float t = distance[1] * stride;
    float r = distance[2] * stride;
    float b = distance[3] * stride;
    
    float x1 = cx - l;
    float y1 = cy - t;
    float x2 = cx + r;
    float y2 = cy + b;
    
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
    
    for (int i = 0; i < 5; i++) {
        float dx = kps_distance[i * 2] * stride;
        float dy = kps_distance[i * 2 + 1] * stride;
        
        float kx = cx + dx;
        float ky = cy + dy;
        
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
    spdlog::info("🔧 [OPT] Inicializando SCRFD TensorRT Optimizado");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    // ✅ CRÍTICO: Crear stream con prioridad alta
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    
    // ✅ CRÍTICO: cv_stream debe wrappear el MISMO stream nativo
    cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    
    if (use_gpu_preprocessing) {
        size_t input_size = input_width * input_height * 3 * sizeof(float);
        cudaMalloc(&d_input_buffer, input_size);
        
        size_t resized_size = input_width * input_height * 3;
        cudaMalloc(&d_resized_buffer, resized_size);
        
        spdlog::info("   GPU buffers allocated:");
        spdlog::info("     - input={:.1f}MB", input_size / 1024.0 / 1024.0);
        spdlog::info("     - resized={:.1f}MB", resized_size / 1024.0 / 1024.0);
    }
    
    spdlog::info("✓ [OPT] Detector optimizado listo (sync fixed v2)");
}

FaceDetectorOptimized::~FaceDetectorOptimized() {
    for (int i = 0; i < 10; i++) {
        if (buffers[i]) cudaFree(buffers[i]);
    }
    if (d_input_buffer) cudaFree(d_input_buffer);
    if (d_resized_buffer) cudaFree(d_resized_buffer);
    if (stream) cudaStreamDestroy(stream);
}

// ==================== LOAD ENGINE (sin cambios) ====================

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

// ==================== PREPROCESSING (CPU - sin cambios) ====================

cv::Mat FaceDetectorOptimized::preprocess_cpu(const cv::Mat& img) {
    cv::Mat resized, normalized;
    cv::resize(img, resized, cv::Size(input_width, input_height));
    
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(normalized, CV_32F);
    normalized = (normalized - 127.5f) / 128.0f;
    
    return normalized;
}

// ==================== PREPROCESSING (GPU - FIXED) ====================

void FaceDetectorOptimized::preprocess_gpu(const cv::Mat& img) {
    /* ✅ CORRECCIÓN CRÍTICA:
     * - Usar SOLO cv_stream para TODAS las operaciones
     * - cv_stream wrappea el stream nativo, garantizando orden de ejecución
     * - NO sincronizar aquí - dejar que el pipeline fluya
     */
    
    // 1. Upload to GPU (usa cv_stream)
    gpu_input.upload(img, cv_stream);
    
    // 2. Resize (usa cv_stream)
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height), 
                     0, 0, cv::INTER_LINEAR, cv_stream);
    
    // 3. ✅ CORRECCIÓN: Obtener el stream nativo desde cv_stream
    cudaStream_t native_stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    
    // 4. Copy resized image (usa el mismo stream)
    cudaMemcpyAsync(
        d_resized_buffer, 
        gpu_resized.data, 
        input_width * input_height * 3,
        cudaMemcpyDeviceToDevice, 
        native_stream  // ✅ Mismo stream = orden garantizado
    );
    
    // 5. Normalize using CUDA kernel (usa el mismo stream)
    cuda_normalize_imagenet(
        static_cast<const unsigned char*>(d_resized_buffer),
        static_cast<float*>(buffers[input_index]),
        input_width, 
        input_height, 
        native_stream  // ✅ Mismo stream = orden garantizado
    );
    
    // ❌ NO sincronizar aquí - dejar fluir el pipeline
}

// ==================== DETECTION (FIXED) ====================

std::vector<Detection> FaceDetectorOptimized::detect(const cv::Mat& img) {
    cv::Size orig_size = img.size();
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (use_gpu_preprocessing) {
        preprocess_gpu(img);
        
        // ✅ CRÍTICO: Sincronizar UNA VEZ después de todas las operaciones GPU
        // Esto garantiza que gpu_input.upload(), resize(), memcpy() y normalize()
        // hayan terminado antes de que TensorRT acceda a los datos
        cudaStreamSynchronize(stream);
        
    } else {
        cv::Mat input_blob = preprocess_cpu(img);
        
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
        
        // Sincronizar después de memcpy
        cudaStreamSynchronize(stream);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = 
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // ✅ TensorRT inference (ya sincronizado)
    bool ok = context->enqueueV3(stream);
    if (!ok) {
        spdlog::error("❌ TensorRT enqueue failed");
        return {};
    }
    
    // ✅ Sincronizar después de inference
    cudaStreamSynchronize(stream);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = 
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Download outputs (async)
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
    
    // ✅ Sincronizar después de copiar outputs
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

// ==================== POSTPROCESSING (sin cambios) ====================

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
    
    struct StrideConfig {
        int feat_h, feat_w, stride;
        int score_idx, bbox_idx, kps_idx;
    };
    
    std::vector<StrideConfig> strides = {
        {80, 80, 8,  0, 1, 2},
        {40, 40, 16, 3, 4, 5},
        {20, 20, 32, 6, 7, 8}
    };
    
    for (const auto& cfg : strides) {
        const float* scores_data = outputs[cfg.score_idx].data();
        const float* bbox_data = outputs[cfg.bbox_idx].data();
        const float* kps_data = outputs[cfg.kps_idx].data();
        
        auto anchors = generate_anchors_scrfd(cfg.feat_h, cfg.feat_w, 
                                             cfg.stride, num_anchors);
        
        int total_anchors = static_cast<int>(anchors.size());
        
        for (int i = 0; i < total_anchors; i++) {
            float score = scores_data[i];
            
            if (score < conf_threshold) continue;
            
            const float* bbox = &bbox_data[i * 4];
            cv::Rect box = distance2bbox(anchors[i], bbox, scale_x, scale_y);
            
            if (box.width < MIN_FACE_SIZE || box.height < MIN_FACE_SIZE) continue;
            if (box.width > orig_size.width * MAX_FACE_RATIO || 
                box.height > orig_size.height * MAX_FACE_RATIO) continue;
            
            box.x = std::max(0, std::min(box.x, orig_size.width - 1));
            box.y = std::max(0, std::min(box.y, orig_size.height - 1));
            box.width = std::min(box.width, orig_size.width - box.x);
            box.height = std::min(box.height, orig_size.height - box.y);
            
            float aspect = static_cast<float>(box.width) / box.height;
            if (aspect < MIN_ASPECT_RATIO || aspect > MAX_ASPECT_RATIO) continue;
            
            Detection det;
            det.confidence = score;
            det.box = box;
            
            const float* kps = &kps_data[i * 10];
            distance2kps(anchors[i], kps, det.landmarks, scale_x, scale_y);
            
            detections.push_back(det);
        }
    }
    
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