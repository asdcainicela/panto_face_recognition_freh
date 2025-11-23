// ============= src/recognizer.cpp =============
#include "recognition/recognizer.hpp"
#include "detection/cuda_kernels.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <cmath>
#include <iostream>

// ... resto del código sin cambios


// ==================== CONSTRUCTOR/DESTRUCTOR ====================

FaceRecognizer::FaceRecognizer(const std::string& engine_path, bool gpu_preproc)
    : use_gpu_preprocessing(gpu_preproc), d_input(nullptr), 
      d_output(nullptr), d_resized(nullptr)
{
    spdlog::info("🎭 Inicializando ArcFace Face Recognizer");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    // Create CUDA stream
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers
    // Allocate GPU buffers
    size_t input_size = 1 * 3 * input_height * input_width * sizeof(float);
    size_t output_size = 1 * embedding_size * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // ---------- LIGAR TENSORES DESPUÉS DE ALLOCAR ----------
    context->setTensorAddress("input.1", d_input);
    context->setTensorAddress("683", d_output);
    // --------------------------------------------------------
    
    if (use_gpu_preprocessing) {
        size_t resized_size = input_height * input_width * 3;
        cudaMalloc(&d_resized, resized_size);
    
        spdlog::info("   GPU buffers: input={:.1f}KB, output={:.1f}KB",
                    input_size / 1024.0, output_size / 1024.0);
    }
    
    spdlog::info("✓ ArcFace Recognizer ready (embedding size: {})", embedding_size);
}

FaceRecognizer::~FaceRecognizer() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_resized) cudaFree(d_resized);
    if (stream) cudaStreamDestroy(stream);
}

// ==================== LOAD ENGINE ====================

bool FaceRecognizer::loadEngine(const std::string& engine_path) {
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
    
    
    spdlog::info("   Engine loaded: {}x{} -> {}", 
                input_width, input_height, embedding_size);
    
    return true;
}

// ==================== PREPROCESSING ====================

cv::Mat FaceRecognizer::preprocess_cpu(const cv::Mat& face) {
    cv::Mat resized, rgb, normalized;
    
    // Resize to 112x112
    cv::resize(face, resized, cv::Size(input_width, input_height));
    
    // BGR -> RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Normalize: (pixel - 127.5) / 128.0
    rgb.convertTo(normalized, CV_32F);
    normalized = (normalized - 127.5f) / 128.0f;
    
    return normalized;
}

void FaceRecognizer::preprocess_gpu(const cv::Mat& face) {
    // Upload to GPU
    gpu_input.upload(face, cv_stream);
    
    // Resize
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height),
                     0, 0, cv::INTER_LINEAR, cv_stream);
    
    // Copy to temporary buffer
    cudaMemcpyAsync(d_resized, gpu_resized.data, 
                   input_width * input_height * 3,
                   cudaMemcpyDeviceToDevice, stream);
    
    // Normalize using CUDA kernel
    cuda_normalize_imagenet(
        static_cast<const unsigned char*>(d_resized),
        static_cast<float*>(d_input),
        input_width, input_height, stream
    );
}

// ==================== EXTRACT EMBEDDING ====================

std::vector<float> FaceRecognizer::extract_embedding(const cv::Mat& face) {
    if (face.empty()) {
        spdlog::warn("Empty face image");
        return std::vector<float>(embedding_size, 0.0f);
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Preprocessing
    if (use_gpu_preprocessing) {
        preprocess_gpu(face);
    } else {
        cv::Mat input_blob = preprocess_cpu(face);
        
        // Convert HWC -> CHW
        std::vector<cv::Mat> channels(3);
        cv::split(input_blob, channels);
        
        size_t single_channel = input_height * input_width;
        std::vector<float> input_data(3 * single_channel);
        
        for (int c = 0; c < 3; c++) {
            std::memcpy(input_data.data() + c * single_channel,
                       channels[c].data, single_channel * sizeof(float));
        }
        
        cudaMemcpyAsync(d_input, input_data.data(),
                       input_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = 
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    /*
    // ---------- DEBUG: ANTES DE ENQUEUE ----------
    std::cout << "[DEBUG] d_input: " << d_input << std::endl;
    std::cout << "[DEBUG] d_output: " << d_output << std::endl;
    std::cout << "[DEBUG] stream: " << stream << std::endl;

    int nb = engine->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        //void* ptr = context->getTensorAddress(name);
        const void* ptr = context->getTensorAddress(name);
        //std::cout << "[DEBUG] Tensor '" << name << "' -> " << ptr << std::endl;
        std::cout << "[DEBUG] Tensor '" << name << "' -> " << ptr << std::endl;
    }
    */
    // ---------------------------------------------
    
    // Inference
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = 
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Get output
    std::vector<float> embedding(embedding_size);
    cudaMemcpyAsync(embedding.data(), d_output,
                   embedding_size * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // L2 normalize
    l2_normalize(embedding);
    
    last_profile.total_ms = last_profile.preprocess_ms + last_profile.inference_ms;
    
    return embedding;
}

// ==================== COMPARISON ====================

float FaceRecognizer::compare(const std::vector<float>& emb1, 
                              const std::vector<float>& emb2) 
{
    if (emb1.size() != emb2.size()) {
        spdlog::error("Embedding size mismatch: {} vs {}", emb1.size(), emb2.size());
        return 0.0f;
    }
    
    // Cosine similarity (assumes normalized embeddings)
    float dot_product = 0.0f;
    for (size_t i = 0; i < emb1.size(); i++) {
        dot_product += emb1[i] * emb2[i];
    }
    
    // Clamp to [-1, 1] (numerical stability)
    return std::max(-1.0f, std::min(1.0f, dot_product));
}

void FaceRecognizer::l2_normalize(std::vector<float>& embedding) {
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0) {
        for (float& val : embedding) {
            val /= norm;
        }
    }
}