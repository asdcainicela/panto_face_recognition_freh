// ============= src/age_gender_predictor.cpp - FIXED =============
#include "age_gender_predictor.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <algorithm>
#include <cmath>

// ==================== AgeGenderResult ====================

std::string AgeGenderResult::to_string() const {
    return std::to_string(age) + " years, " + gender_to_string(gender);
}

std::string gender_to_string(Gender gender) {
    return (gender == Gender::MALE) ? "Male" : "Female";
}

// ==================== AgeGenderPredictor ====================

AgeGenderPredictor::AgeGenderPredictor(const std::string& engine_path, bool gpu_preproc)
    : use_gpu_preprocessing(gpu_preproc), d_input(nullptr), 
      d_output(nullptr), d_resized(nullptr)
{
    spdlog::info("👤 Inicializando Age-Gender Predictor");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("No se pudo cargar TensorRT engine");
    }
    
    cudaStreamCreate(&stream);
    // NO usar cv::cuda::Stream - trabajar directamente con cudaStream_t
    
    // Allocate GPU buffers
    size_t input_size = 1 * 3 * input_height * input_width * sizeof(float);
    size_t output_size = 1 * num_classes * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // Bind tensors
    context->setTensorAddress("pixel_values", d_input);
    context->setTensorAddress("logits", d_output);
    
    if (use_gpu_preprocessing) {
        size_t resized_size = input_height * input_width * 3;
        cudaMalloc(&d_resized, resized_size);
        spdlog::info("   GPU buffers: input={:.1f}KB, output={:.1f}KB",
                    input_size / 1024.0, output_size / 1024.0);
    }
    
    spdlog::info("✓ Age-Gender Predictor ready ({} classes)", num_classes);
}

AgeGenderPredictor::~AgeGenderPredictor() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_resized) cudaFree(d_resized);
    if (stream) cudaStreamDestroy(stream);
}

bool AgeGenderPredictor::loadEngine(const std::string& engine_path) {
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
    
    // Get output size
    const char* output_name = "logits";
    auto dims = engine->getTensorShape(output_name);
    num_classes = dims.d[1];
    
    spdlog::info("   Engine loaded: {}x{} -> {} classes", 
                input_width, input_height, num_classes);
    
    return true;
}

cv::Mat AgeGenderPredictor::preprocess_cpu(const cv::Mat& face) {
    cv::Mat resized, rgb, normalized;
    
    // Resize to 224x224
    cv::resize(face, resized, cv::Size(input_width, input_height));
    
    // BGR -> RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Normalize ImageNet: (pixel / 255.0 - mean) / std
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    
    cv::subtract(normalized, mean, normalized);
    cv::divide(normalized, std, normalized);
    
    return normalized;
}

void AgeGenderPredictor::preprocess_gpu(const cv::Mat& face) {
    // Upload to GPU
    gpu_input.upload(face);
    
    // Resize
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height),
                     0, 0, cv::INTER_LINEAR);
    
    // Copy resized image to d_resized buffer (GPU -> GPU)
    cudaMemcpyAsync(d_resized, gpu_resized.data, 
                   input_width * input_height * 3,
                   cudaMemcpyDeviceToDevice, stream);
    
    // Download to CPU for normalization (GPU -> CPU)
    std::vector<unsigned char> temp(input_width * input_height * 3);
    cudaMemcpyAsync(temp.data(), d_resized,
                   input_width * input_height * 3,
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // BGR -> RGB + normalize ImageNet on CPU
    std::vector<float> normalized(input_width * input_height * 3);
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    
    size_t plane_size = input_width * input_height;
    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            int idx = y * input_width + x;
            int pixel_idx = idx * 3;
            
            // BGR input
            float b = temp[pixel_idx + 0] / 255.0f;
            float g = temp[pixel_idx + 1] / 255.0f;
            float r = temp[pixel_idx + 2] / 255.0f;
            
            // Output: CHW format, RGB order, normalized
            normalized[0 * plane_size + idx] = (r - mean[0]) / std[0];  // R
            normalized[1 * plane_size + idx] = (g - mean[1]) / std[1];  // G
            normalized[2 * plane_size + idx] = (b - mean[2]) / std[2];  // B
        }
    }
    
    // Upload normalized data to GPU (CPU -> GPU)
    cudaMemcpyAsync(d_input, normalized.data(),
                   normalized.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
}

AgeGenderResult AgeGenderPredictor::predict(const cv::Mat& face) {
    if (face.empty()) {
        spdlog::warn("Empty face image");
        return AgeGenderResult{0, Gender::MALE, 0.0f, 0.0f};
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

AgeGenderResult AgeGenderPredictor::postprocess(const std::vector<float>& logits) {
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
    
    // Gender: primeras 2 clases [Female, Male]
    float female_prob = probs[0];
    float male_prob = probs[1];
    
    Gender gender = (male_prob > female_prob) ? Gender::MALE : Gender::FEMALE;
    float gender_confidence = std::max(female_prob, male_prob);
    
    // Age: siguientes clases (usualmente ~100 clases para 0-100 años)
    int age_start_idx = 2;
    int max_age_idx = age_start_idx;
    float max_age_prob = probs[age_start_idx];
    
    for (size_t i = age_start_idx; i < probs.size(); i++) {
        if (probs[i] > max_age_prob) {
            max_age_prob = probs[i];
            max_age_idx = i;
        }
    }
    
    // La edad es el índice - 2 (porque las primeras 2 son género)
    int age = max_age_idx - age_start_idx;
    
    AgeGenderResult result;
    result.age = age;
    result.gender = gender;
    result.age_confidence = max_age_prob;
    result.gender_confidence = gender_confidence;
    
    return result;
}

std::vector<AgeGenderResult> AgeGenderPredictor::predict_batch(
    const std::vector<cv::Mat>& faces) 
{
    std::vector<AgeGenderResult> results;
    results.reserve(faces.size());
    
    for (const auto& face : faces) {
        results.push_back(predict(face));
    }
    
    return results;
}