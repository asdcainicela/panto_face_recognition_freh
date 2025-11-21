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

// AgeGenderResult
std::string AgeGenderResult::to_string() const {
    return std::to_string(age) + " years, " + gender_to_string(gender);
}

std::string gender_to_string(Gender gender) {
    return (gender == Gender::MALE) ? "Male" : "Female";
}

// AgeGenderPredictor

AgeGenderPredictor::AgeGenderPredictor(const std::string& engine_path, bool gpu_preproc)
    : use_gpu_preprocessing(gpu_preproc), d_input(nullptr),
      d_output(nullptr), d_resized(nullptr)
{
    spdlog::info("inicializando age-gender predictor");
    spdlog::info("gpu preprocessing: {}", gpu_preproc ? "enabled" : "disabled");

    if (!loadEngine(engine_path)) {
        throw std::runtime_error("cannot load engine");
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
        spdlog::info("gpu buffers: input={}KB output={}KB", input_size_bytes / 1024.0, output_size_bytes / 1024.0);
    }

    spdlog::info("ready ({} classes)", num_classes);
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
        spdlog::error("cannot open engine: {}", engine_path);
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

    // try to get output shape safely
    const char* output_name = "logits";
    try {
        auto dims = engine->getTensorShape(output_name);
        if (dims.nbDims >= 2) {
            num_classes = dims.d[dims.nbDims - 1];
        } else if (dims.nbDims == 1) {
            num_classes = dims.d[0];
        } else {
            num_classes = 2; // fallback
        }
    } catch (...) {
        num_classes = 2;
    }

    spdlog::info("engine loaded: {}x{} -> {} classes", input_width, input_height, num_classes);
    return true;
}

cv::Mat AgeGenderPredictor::preprocess_cpu(const cv::Mat& face) {
    cv::Mat resized, rgb, normalized;
    cv::resize(face, resized, cv::Size(input_width, input_height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(normalized, mean, normalized);
    cv::divide(normalized, std, normalized);
    return normalized;
}

void AgeGenderPredictor::preprocess_gpu(const cv::Mat& face) {
    if (face.empty()) {
        throw std::runtime_error("empty face for gpu preprocess");
    }

    cv::cuda::GpuMat d_src;
    d_src.upload(face);

    cv::cuda::GpuMat d_resized_uc;
    d_resized_uc.create(input_height, input_width, CV_8UC3);
    cv::cuda::resize(d_src, d_resized_uc, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    cv::cuda::GpuMat d_rgb_uc;
    d_rgb_uc.create(input_height, input_width, CV_8UC3);
    cv::cuda::cvtColor(d_resized_uc, d_rgb_uc, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat d_rgb_f;
    d_rgb_f.create(input_height, input_width, CV_32FC3);
    d_rgb_uc.convertTo(d_rgb_f, CV_32F, 1.0 / 255.0);

    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(d_rgb_f, channels); // channels: R,G,B each CV_32F

    float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    float std_vals[3]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        cv::cuda::GpuMat tmp;
        cv::cuda::multiply(channels[c], cv::Scalar(1.0f / std_vals[c]), tmp);
        cv::cuda::subtract(tmp, cv::Scalar(mean_vals[c] / std_vals[c]), channels[c]);
    }

    // download HWC float to host
    cv::Mat host_hwc(input_height, input_width, CV_32FC3);
    d_rgb_f.upload(cv::Mat()); // noop to keep API usage consistent
    cv::cuda::GpuMat d_merged;
    cv::cuda::merge(channels, d_merged);
    d_merged.download(host_hwc);

    // reorder HWC -> CHW on host
    size_t plane = static_cast<size_t>(input_height) * input_width;
    std::vector<float> host_chw(3 * plane);
    std::vector<cv::Mat> host_channels(3);
    cv::split(host_hwc, host_channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(host_chw.data() + c * plane, host_channels[c].data, plane * sizeof(float));
    }

    // upload to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, host_chw.data(), host_chw.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

AgeGenderResult AgeGenderPredictor::predict(const cv::Mat& face) {
    if (face.empty()) {
        spdlog::warn("empty face image");
        return AgeGenderResult{0, Gender::MALE, 0.0f, 0.0f};
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
        CUDA_CHECK(cudaMemcpyAsync(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_profile.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = context->enqueueV3(stream);
    if (!ok) {
        spdlog::error("enqueue failed");
        throw std::runtime_error("enqueue failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t2 = std::chrono::high_resolution_clock::now();
    last_profile.inference_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::vector<float> logits(num_classes);
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), d_output, num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    last_profile.total_ms = last_profile.preprocess_ms + last_profile.inference_ms;
    return postprocess(logits);
}

AgeGenderResult AgeGenderPredictor::postprocess(const std::vector<float>& logits) {
    std::vector<float> expv(logits.size());
    float maxv = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        expv[i] = std::exp(logits[i] - maxv);
        sum += expv[i];
    }
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) probs[i] = expv[i] / sum;

    float female_prob = probs.size() > 0 ? probs[0] : 0.0f;
    float male_prob   = probs.size() > 1 ? probs[1] : 0.0f;
    Gender gender = (male_prob > female_prob) ? Gender::MALE : Gender::FEMALE;
    float gender_conf = std::max(female_prob, male_prob);

    int age_start = 2;
    int max_idx = age_start;
    float max_prob = (probs.size() > (size_t)age_start) ? probs[age_start] : 0.0f;
    for (size_t i = age_start; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = static_cast<int>(i);
        }
    }
    int age = max_idx - age_start;

    AgeGenderResult r;
    r.age = age;
    r.gender = gender;
    r.age_confidence = max_prob;
    r.gender_confidence = gender_conf;
    return r;
}

std::vector<AgeGenderResult> AgeGenderPredictor::predict_batch(const std::vector<cv::Mat>& faces) {
    std::vector<AgeGenderResult> out;
    out.reserve(faces.size());
    for (const auto& f : faces) out.push_back(predict(f));
    return out;
}
