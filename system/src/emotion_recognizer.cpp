// ============= src/emotion_recognizer.cpp =============
#include "emotion_recognizer.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <algorithm>
#include <cmath>

// ==================== EmotionResult ====================

std::string EmotionResult::to_string() const {
    return emotion_to_string(emotion) + 
           " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";
}

std::string emotion_to_string(Emotion emotion) {
    switch (emotion) {
        case Emotion::NEUTRAL:    return "Neutral";
        case Emotion::HAPPINESS:  return "Happy";
        case Emotion::SURPRISE:   return "Surprised";
        case Emotion::SADNESS:    return "Sad";
        case Emotion::ANGER:      return "Angry";
        case Emotion::DISGUST:    return "Disgusted";
        case Emotion::FEAR:       return "Fearful";
        case Emotion::CONTEMPT:   return "Contemptuous";
        default:                  return "Unknown";
    }
}

// ==================== EmotionRecognizer ====================

EmotionRecognizer::EmotionRecognizer(const std::string& engine_path, bool gpu_preproc)
    : use_gpu_preprocessing(gpu_preproc), d_input(nullptr), 
      d_output(nullptr), d_resized(nullptr)
{
    spdlog::info("😊 Inicializando FER+ Emotion Recognizer");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    cudaStreamCreate(&stream);
    
    // Allocate GPU buffers
    size_t input_size = 1 * 1 * input_height * input_width * sizeof(float);
    size_t output_size = 1 * num_classes * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // Bind tensors
    context->setTensorAddress("Input3", d_input);
    context->setTensorAddress("Plus692_Output_0", d_output);
    
    if (use_gpu_preprocessing) {
        size_t resized_size = input_height * input_width;
        cudaMalloc(&d_resized, resized_size);
        spdlog::info("   GPU buffers: input={:.1f}KB, output={:.1f}KB",
                    input_size / 1024.0, output_size / 1024.0);
    }
    
    spdlog::info("✓ Emotion Recognizer ready (8 emotions)");
}

EmotionRecognizer::~EmotionRecognizer() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_resized) cudaFree(d_resized);
    if (stream) cudaStreamDestroy(stream);
}

bool EmotionRecognizer::loadEngine(const std::string& engine_path) {
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
    
    spdlog::info("   Engine loaded: {}x{} -> {} classes", 
                input_width, input_height, num_classes);
    
    return true;
}

cv::Mat EmotionRecognizer::preprocess_cpu(const cv::Mat& face) {
    cv::Mat gray, resized, normalized;
    
    // Convert to grayscale
    cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
    
    // Resize to 64x64
    cv::resize(gray, resized, cv::Size(input_width, input_height));
    
    // Normalize [0, 1]
    resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    return normalized;
}

void EmotionRecognizer::preprocess_gpu(const cv::Mat& face) {
    // Upload to GPU
    gpu_input.upload(face);
    
    // BGR -> GRAY
    cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);
    
    // Resize
    cv::cuda::resize(gpu_gray, gpu_resized, 
                     cv::Size(input_width, input_height),
                     0, 0, cv::INTER_LINEAR);
    
    // Download to d_resized (as uchar)
    cudaMemcpyAsync(d_resized, gpu_resized.data, 
                   input_width * input_height,
                   cudaMemcpyDeviceToDevice, stream);
    
    // Normalize to [0, 1] using custom kernel (simple divide by 255)
    // For now, do it on CPU (later optimize with kernel)
    std::vector<unsigned char> temp(input_width * input_height);
    cudaMemcpyAsync(temp.data(), d_resized,
                   input_width * input_height,
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<float> normalized(input_width * input_height);
    for (size_t i = 0; i < temp.size(); i++) {
        normalized[i] = temp[i] / 255.0f;
    }
    
    cudaMemcpyAsync(d_input, normalized.data(),
                   normalized.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
}

EmotionResult EmotionRecognizer::predict(const cv::Mat& face) {
    if (face.empty()) {
        spdlog::warn("Empty face image");
        return EmotionResult{Emotion::NEUTRAL, 0.0f, {}};
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Preprocessing
    if (use_gpu_preprocessing) {
        preprocess_gpu(face);
    } else {
        cv::Mat input_blob = preprocess_cpu(face);
        
        cudaMemcpyAsync(d_input, input_blob.data,
                       input_width * input_height * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = 
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Inference
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = 
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    
    // Get output
    std::vector<float> logits(num_classes);
    cudaMemcpyAsync(logits.data(), d_output,
                   num_classes * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    last_profile.total_ms = last_profile.preprocess_ms + last_profile.inference_ms;
    
    return postprocess(logits);
}

EmotionResult EmotionRecognizer::postprocess(const std::vector<float>& logits) {
    // Softmax
    std::vector<float> exp_vals(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < logits.size(); i++) {
        exp_vals[i] = std::exp(logits[i] - max_logit);
        sum_exp += exp_vals[i];
    }
    
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = exp_vals[i] / sum_exp;
    }
    
    // Find max
    int max_idx = std::distance(probs.begin(), 
                               std::max_element(probs.begin(), probs.end()));
    
    EmotionResult result;
    result.emotion = static_cast<Emotion>(max_idx);
    result.confidence = probs[max_idx];
    result.probabilities = probs;
    
    return result;
}

std::vector<EmotionResult> EmotionRecognizer::predict_batch(
    const std::vector<cv::Mat>& faces) 
{
    std::vector<EmotionResult> results;
    results.reserve(faces.size());
    
    for (const auto& face : faces) {
        results.push_back(predict(face));
    }
    
    return results;
}