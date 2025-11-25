// ============= src/age_gender_predictor.cpp - FAIRFACE COMPLETO =============
#include "age_gender_predictor.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        spdlog::error("cuda error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while (0)

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
    spdlog::info("🎂 Inicializando Age/Gender Predictor (FairFace ResNet-34)");
    spdlog::info("   GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");

    if (!loadEngine(engine_path)) {
        throw std::runtime_error("Cannot load engine");
    }

    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t input_size_bytes = static_cast<size_t>(1) * 3 * input_height * input_width * sizeof(float);
    size_t output_size_bytes = static_cast<size_t>(1) * num_classes * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_size_bytes));

    context->setTensorAddress("pixel_values", d_input);
    context->setTensorAddress("logits", d_output);

    if (use_gpu_preprocessing) {
        size_t resized_bytes = static_cast<size_t>(input_height) * input_width * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_resized, resized_bytes));
        spdlog::info("   GPU buffers: input={:.1f}KB output={:.1f}KB",
                    input_size_bytes / 1024.0, output_size_bytes / 1024.0);
    }

    spdlog::info("✅ Age/Gender Predictor ready");
    spdlog::info("   Input: {}x{}", input_width, input_height);
    spdlog::info("   Output: {} classes (2 gender + 9 age brackets)", num_classes);
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
        spdlog::error("Cannot open engine: {}", engine_path);
        return false;
    }

    file.seekg(0, file.end);
    size_t size = static_cast<size_t>(file.tellg());
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

    // ✅ OBTENER NUM_CLASSES DEL ENGINE
    const char* output_name = "logits";
    try {
        auto dims = engine->getTensorShape(output_name);
        if (dims.nbDims >= 2) {
            num_classes = dims.d[dims.nbDims - 1];
        } else if (dims.nbDims == 1) {
            num_classes = dims.d[0];
        } else {
            num_classes = 11; // ✅ FairFace default
        }
    } catch (...) {
        num_classes = 11; // ✅ FairFace default
    }

    spdlog::info("   Engine loaded: {}x{} -> {} output classes", 
                input_width, input_height, num_classes);
    
    if (num_classes != 11) {
        spdlog::warn("⚠️  Expected 11 classes (FairFace), got {}. Predictions may be incorrect.", num_classes);
    }
    
    return true;
}

cv::Mat AgeGenderPredictor::preprocess_cpu(const cv::Mat& face) {
    cv::Mat resized, rgb, normalized;
    cv::resize(face, resized, cv::Size(input_width, input_height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    // ImageNet normalization
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(normalized, mean, normalized);
    cv::divide(normalized, std, normalized);
    
    return normalized;
}

void AgeGenderPredictor::preprocess_gpu(const cv::Mat& face) {
    if (face.empty()) {
        throw std::runtime_error("Empty face for GPU preprocess");
    }

    // 1. Upload to GPU
    gpu_input.upload(face);
    
    // 2. Resize 224x224 (GPU)
    cv::cuda::resize(gpu_input, gpu_resized, 
                     cv::Size(input_width, input_height), 
                     0, 0, cv::INTER_LINEAR);
    
    // 3. Copy resized image to d_resized buffer
    CUDA_CHECK(cudaMemcpyAsync(
        d_resized, 
        gpu_resized.data, 
        input_width * input_height * 3,
        cudaMemcpyDeviceToDevice, 
        stream
    ));
    
    // 4. CUDA kernel: BGR->RGB + ImageNet normalize + HWC->CHW
    cuda_normalize_age_gender(
        static_cast<const unsigned char*>(d_resized),
        static_cast<float*>(d_input),
        input_width, 
        input_height, 
        stream
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

AgeGenderResult AgeGenderPredictor::predict(const cv::Mat& face) {
    if (face.empty()) {
        spdlog::warn("Empty face image");
        return AgeGenderResult{25, Gender::MALE, 0.0f, 0.0f};
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (use_gpu_preprocessing) {
        preprocess_gpu(face);
    } else {
        cv::Mat input_blob = preprocess_cpu(face);
        std::vector<cv::Mat> channels(3);
        cv::split(input_blob, channels);
        size_t single = static_cast<size_t>(input_height) * input_width;
        std::vector<float> input_data(3 * single);
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_data.data() + c * single, channels[c].data, single * sizeof(float));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_input, input_data.data(), 
                   input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = context->enqueueV3(stream);
    if (!ok) {
        spdlog::error("Enqueue failed");
        throw std::runtime_error("Enqueue failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::vector<float> logits(num_classes);
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_output, 
               num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    last_profile.total_ms = last_profile.preprocess_ms + last_profile.inference_ms;
    return postprocess(logits);
}

AgeGenderResult AgeGenderPredictor::postprocess(const std::vector<float>& logits) {
    // ✅ FORMATO FAIRFACE:
    // Output: [1, 11]
    // Index 0-1: Gender (Male, Female)
    // Index 2-10: Age brackets (9 rangos)
    
    if (logits.size() < 11) {
        spdlog::error("Invalid logits size: {} (expected 11)", logits.size());
        return AgeGenderResult{25, Gender::MALE, 0.0f, 0.0f}; // default
    }
    
    // ============ GENDER PREDICTION ============
    float male_logit = logits[0];
    float female_logit = logits[1];
    
    // Softmax para gender
    float max_gender = std::max(male_logit, female_logit);
    float exp_male = std::exp(male_logit - max_gender);
    float exp_female = std::exp(female_logit - max_gender);
    float sum_gender = exp_male + exp_female;
    
    float male_prob = exp_male / sum_gender;
    float female_prob = exp_female / sum_gender;
    
    Gender gender = (male_prob > female_prob) ? Gender::MALE : Gender::FEMALE;
    float gender_conf = std::max(male_prob, female_prob);
    
    // ============ AGE PREDICTION ============
    // Age logits: indices 2-10 (9 clases)
    std::vector<float> age_logits(logits.begin() + 2, logits.end());
    
    // Softmax para age
    float max_age = *std::max_element(age_logits.begin(), age_logits.end());
    std::vector<float> age_probs(age_logits.size());
    float sum_age = 0.0f;
    
    for (size_t i = 0; i < age_logits.size(); ++i) {
        age_probs[i] = std::exp(age_logits[i] - max_age);
        sum_age += age_probs[i];
    }
    
    for (size_t i = 0; i < age_probs.size(); ++i) {
        age_probs[i] /= sum_age;
    }
    
    // Encontrar clase con mayor probabilidad
    int max_age_idx = 0;
    float max_age_prob = age_probs[0];
    for (size_t i = 1; i < age_probs.size(); ++i) {
        if (age_probs[i] > max_age_prob) {
            max_age_prob = age_probs[i];
            max_age_idx = i;
        }
    }
    
    // Mapeo de clase a edad (punto medio del rango)
    const int age_map[] = {
        1,   // 0-2 years   -> 1
        6,   // 3-9 years   -> 6
        15,  // 10-19 years -> 15
        25,  // 20-29 years -> 25
        35,  // 30-39 years -> 35
        45,  // 40-49 years -> 45
        55,  // 50-59 years -> 55
        65,  // 60-69 years -> 65
        75   // 70+ years   -> 75
    };
    
    int predicted_age = age_map[max_age_idx];
    
    // ✅ DEBUG (solo primeras 3 predicciones)
    static int debug_count = 0;
    if (debug_count < 3) {
        spdlog::info("🔍 Age/Gender Debug:");
        spdlog::info("  Gender: Male={:.3f}, Female={:.3f} -> {} ({:.1f}%)",
                    male_prob, female_prob, gender_to_string(gender), gender_conf * 100);
        spdlog::info("  Age class: {} (prob={:.3f})", max_age_idx, max_age_prob);
        spdlog::info("  Age brackets probabilities:");
        const char* age_labels[] = {"0-2", "3-9", "10-19", "20-29", "30-39", 
                                   "40-49", "50-59", "60-69", "70+"};
        for (size_t i = 0; i < age_probs.size(); ++i) {
            spdlog::info("    {} years: {:.3f}", age_labels[i], age_probs[i]);
        }
        spdlog::info("  Final prediction: {} years, {}", predicted_age, gender_to_string(gender));
        debug_count++;
    }
    
    AgeGenderResult result;
    result.age = predicted_age;
    result.gender = gender;
    result.age_confidence = max_age_prob;
    result.gender_confidence = gender_conf;
    
    return result;
}

std::vector<AgeGenderResult> AgeGenderPredictor::predict_batch(const std::vector<cv::Mat>& faces) {
    std::vector<AgeGenderResult> out;
    out.reserve(faces.size());
    for (const auto& f : faces) {
        out.push_back(predict(f));
    }
    return out;
}