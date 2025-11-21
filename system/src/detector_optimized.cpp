// ============= src/detector_optimized.cpp =============
/*
 * SCRFD Face Detector - TensorRT Optimized Implementation
 * 
 * CARACTERÍSTICAS:
 * - Detección de rostros usando SCRFD (Sample and Computation Redistribution for Face Detection)
 * - Aceleración TensorRT con soporte FP16
 * - Preprocessing GPU opcional (CUDA kernels custom)
 * - Multi-scale detection: strides 8, 16, 32 (80x80, 40x40, 20x20 feature maps)
 * - NMS (Non-Maximum Suppression) con IoU + IoM (Intersection over Minimum)
 * 
 * ARQUITECTURA DEL MODELO:
 * - Input: 640x640x3 (BGR -> RGB, normalizado: (x-127.5)/128.0)
 * - Output: 9 tensors secuenciales por stride:
 *   [stride8_scores, stride8_bbox, stride8_kps,
 *    stride16_scores, stride16_bbox, stride16_kps,
 *    stride32_scores, stride32_bbox, stride32_kps]
 * 
 * FORMATO DE OUTPUTS:
 * - Scores: [N] confidence por anchor (N = feat_h * feat_w * num_anchors)
 * - BBox: [N, 4] distancias (left, top, right, bottom) desde anchor center
 * - Keypoints: [N, 10] offsets para 5 landmarks (2 coords cada uno)
 * 
 * ANCHORS:
 * - 2 anchors por posición espacial (num_anchors = 2)
 * - Generados con offset 0.5: center = (j + 0.5) * stride
 * - Total: 12800 (stride 8) + 3200 (stride 16) + 800 (stride 32) = 16800
 * 
 * DECODIFICACIÓN:
 * - BBox: x1 = cx - l*stride, y1 = cy - t*stride
 *         x2 = cx + r*stride, y2 = cy + b*stride
 * - Keypoints: kx = cx + dx*stride, ky = cy + dy*stride
 * - Escalado: multiplicar por (orig_w/640, orig_h/640)
 * 
 * VALIDACIONES:
 * - Tamaño mínimo: 20x20 pixels
 * - Tamaño máximo: 90% de la imagen
 * - Aspect ratio: 0.4 - 2.5
 * - Clipping a límites de imagen
 * 
 * NMS:
 * - Threshold IoU: configurable (default 0.4)
 * - Threshold IoM: 0.8 (para eliminar cajas muy contenidas)
 * - Sort by confidence descendente
 * 
 * OPTIMIZACIONES:
 * - CUDA streams para operaciones asíncronas
 * - GPU preprocessing con kernels custom (BGR->RGB + normalize)
 * - TensorRT FP16 inference
 * - Memory pooling (buffers pre-allocated)
 * 
 * PERFORMANCE (Jetson Orin):
 * - 1920x1080: ~15-20ms total (GPU preproc ON)
 * - Breakdown: preproc ~2-3ms, inference ~8-10ms, postproc ~3-5ms
 * 
 * AUTOR: PANTO System
 * FECHA: 2025
 */

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
    
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
    
    cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    
    if (use_gpu_preprocessing) {
        size_t input_size = input_width * input_height * 3 * sizeof(float);
        cudaMalloc(&d_input_buffer, input_size);
        
        // ✅ AÑADIDO: Allocate d_resized_buffer (unsigned char, no float)
        size_t resized_size = input_width * input_height * 3;  // 640*640*3 bytes
        cudaMalloc(&d_resized_buffer, resized_size);
        
        spdlog::info("   GPU buffers allocated:");
        spdlog::info("     - d_input_buffer: {:.1f}MB", input_size / 1024.0 / 1024.0);
        spdlog::info("     - d_resized_buffer: {:.1f}MB", resized_size / 1024.0 / 1024.0);
    }
    
    spdlog::info("✓ [OPT] Detector optimizado listo");
}

FaceDetectorOptimized::~FaceDetectorOptimized() {
    for (int i = 0; i < 10; i++) {
        if (buffers[i]) cudaFree(buffers[i]);
    }
    if (d_input_buffer) cudaFree(d_input_buffer);
    if (d_resized_buffer) cudaFree(d_resized_buffer);  // ✅ AÑADIDO
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
    
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(normalized, CV_32F);
    normalized = (normalized - 127.5f) / 128.0f;
    
    return normalized;
}


// ==================== PREPROCESS GPU - FIXED ====================
void FaceDetectorOptimized::preprocess_gpu(const cv::Mat& img) {
    // 1. Upload to GPU (OpenCV stream)
    gpu_input.upload(img, cv_stream);
    
    // 2. Resize (OpenCV stream)
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height), 
                     0, 0, cv::INTER_LINEAR, cv_stream);
    
    // ✅ FIX 1: Sincronizar OpenCV stream ANTES de usar CUDA nativo
    cv_stream.waitForCompletion();
    
    // ✅ FIX 2: Verificar que gpu_resized tiene datos válidos
    if (gpu_resized.empty() || gpu_resized.cols != input_width || gpu_resized.rows != input_height) {
        spdlog::error("❌ GPU resize failed: {}x{}", gpu_resized.cols, gpu_resized.rows);
        throw std::runtime_error("GPU resize produced invalid output");
    }
    
    // 3. Copy resized image to d_resized_buffer (CUDA stream nativo)
    cudaError_t copy_err = cudaMemcpyAsync(
        d_resized_buffer,  // ✅ CORREGIDO: usar d_resized_buffer
        gpu_resized.data, 
        input_width * input_height * 3,  // 640*640*3 bytes
        cudaMemcpyDeviceToDevice, 
        stream
    );
    
    if (copy_err != cudaSuccess) {
        spdlog::error("❌ cudaMemcpy failed: {}", cudaGetErrorString(copy_err));
        throw std::runtime_error("GPU memory copy failed");
    }
    
    // ✅ FIX 3: Sincronizar ANTES del kernel (crítico)
    cudaStreamSynchronize(stream);
    
    // 4. Normalize using CUDA kernel
    cuda_normalize_imagenet(
        static_cast<const unsigned char*>(d_resized),
        static_cast<float*>(buffers[input_index]),
        input_width, 
        input_height, 
        stream
    );
    
    // ✅ FIX 4: Verificar errores del kernel
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        spdlog::error("❌ Kernel launch failed: {}", cudaGetErrorString(kernel_err));
        throw std::runtime_error("CUDA kernel failed");
    }
    
    // ✅ FIX 5: Sincronizar después del kernel
    cudaStreamSynchronize(stream);
    
    // ✅ FIX 6 (OPCIONAL): Verificar output del kernel (solo para debug)
    static int verify_count = 0;
    if (verify_count++ < 1) {
        std::vector<float> debug_data(100);
        cudaMemcpy(debug_data.data(), buffers[input_index], 
                   100 * sizeof(float), cudaMemcpyDeviceToHost);
        
        float min_val = *std::min_element(debug_data.begin(), debug_data.end());
        float max_val = *std::max_element(debug_data.begin(), debug_data.end());
        
        spdlog::info("🔍 Tensor normalizado: min={:.4f}, max={:.4f}", min_val, max_val);
        
        // ✅ Valores esperados: [-1.0, +1.0] aproximadamente
        if (min_val < -2.0f || max_val > 2.0f) {
            spdlog::error("❌ Valores fuera de rango esperado [-1, +1]!");
        }
        
        // Mostrar primeros 10 valores
        spdlog::info("🔍 Primeros 10 valores:");
        for (int i = 0; i < 10; i++) {
            spdlog::info("  [{}] = {:.4f}", i, debug_data[i]);
        }
    }
}

// ==================== DETECTION ====================

std::vector<Detection> FaceDetectorOptimized::detect(const cv::Mat& img) {
    cv::Size orig_size = img.size();
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (use_gpu_preprocessing) {
        preprocess_gpu(img);
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
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = 
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = 
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    
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
    
    struct StrideConfig {
        int feat_h, feat_w, stride;
        int score_idx, bbox_idx, kps_idx;
    };
    
    std::vector<StrideConfig> strides = {
        {80, 80, 8,  0, 1, 2},   // Stride 8:  outputs[0,1,2] (scores, bbox, kps)
        {40, 40, 16, 3, 4, 5},   // Stride 16: outputs[3,4,5]
        {20, 20, 32, 6, 7, 8}    // Stride 32: outputs[6,7,8]
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