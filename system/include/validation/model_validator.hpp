// ============= include/validation/model_validator.hpp =============
#pragma once
#include <string>
#include <vector>
#include <map>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include "core/tensorrt_logger.hpp"  

struct TensorInfo {
    std::string name;
    nvinfer1::TensorIOMode mode;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t size_bytes;

    std::string dims_to_string() const {
        std::string result = "[";
        for (int i = 0; i < dims.nbDims; i++) {
            result += std::to_string(dims.d[i]);
            if (i < dims.nbDims - 1) result += ", ";
        }
        result += "]";
        return result;
    }

    std::string dtype_to_string() const {
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT: return "FLOAT32";
            case nvinfer1::DataType::kHALF: return "FLOAT16";
            case nvinfer1::DataType::kINT8: return "INT8";
            case nvinfer1::DataType::kINT32: return "INT32";
            default: return "UNKNOWN";
        }
    }
};

struct ModelInfo {
    std::string path;
    bool exists;
    bool valid_engine;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    size_t total_size_mb;

    void print() const {
        spdlog::info("Model: {}", path);
        spdlog::info("Exists: {}", exists);
        spdlog::info("Valid engine: {}", valid_engine);
        spdlog::info("Size (MB): {}", total_size_mb);

        spdlog::info("Inputs: {}", inputs.size());
        for (const auto& tensor : inputs) {
            spdlog::info("  {}", tensor.name);
            spdlog::info("    Shape: {}", tensor.dims_to_string());
            spdlog::info("    Type: {}", tensor.dtype_to_string());
        }

        spdlog::info("Outputs: {}", outputs.size());
        for (const auto& tensor : outputs) {
            spdlog::info("  {}", tensor.name);
            spdlog::info("    Shape: {}", tensor.dims_to_string());
            spdlog::info("    Type: {}", tensor.dtype_to_string());
        }
    }
};

class ModelValidator {
private:
    panto::TensorRTLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;

    bool file_exists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }

    size_t get_file_size_mb(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.good()) return 0;
        return static_cast<size_t>(file.tellg()) / (1024 * 1024);
    }

    ModelInfo inspect_engine(const std::string& path) {
        ModelInfo info;
        info.path = path;
        info.exists = file_exists(path);
        info.valid_engine = false;
        info.total_size_mb = 0;

        if (!info.exists) {
            return info;
        }

        info.total_size_mb = get_file_size_mb(path);

        std::ifstream file(path, std::ios::binary);
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        if (!runtime) {
            runtime.reset(nvinfer1::createInferRuntime(logger));
        }

        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), size)
        );

        if (!engine) {
            return info;
        }

        info.valid_engine = true;

        int nb_tensors = engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; i++) {
            const char* name = engine->getIOTensorName(i);

            TensorInfo tensor;
            tensor.name = name;
            tensor.mode = engine->getTensorIOMode(name);
            tensor.dims = engine->getTensorShape(name);
            tensor.dtype = engine->getTensorDataType(name);

            size_t s = 1;
            for (int j = 0; j < tensor.dims.nbDims; j++) {
                s *= tensor.dims.d[j];
            }

            switch (tensor.dtype) {
                case nvinfer1::DataType::kFLOAT:
                case nvinfer1::DataType::kINT32:
                    tensor.size_bytes = s * 4;
                    break;
                case nvinfer1::DataType::kHALF:
                    tensor.size_bytes = s * 2;
                    break;
                case nvinfer1::DataType::kINT8:
                    tensor.size_bytes = s;
                    break;
                default:
                    tensor.size_bytes = s * 4;
            }

            if (tensor.mode == nvinfer1::TensorIOMode::kINPUT) {
                info.inputs.push_back(tensor);
            } else {
                info.outputs.push_back(tensor);
            }
        }

        return info;
    }

public:
    ModelValidator() {
        runtime.reset(nvinfer1::createInferRuntime(logger));
    }

    bool validate_detector(const std::string& path) {
        auto info = inspect_engine(path);
        info.print();

        if (!info.valid_engine) {
            spdlog::error("Invalid detector engine");
            return false;
        }

        if (info.inputs.size() != 1) {
            spdlog::error("Expected 1 input, got {}", info.inputs.size());
            return false;
        }

        if (info.outputs.size() != 9) {
            spdlog::warn("Expected 9 outputs, got {}", info.outputs.size());
        }

        auto& input = info.inputs[0];
        if (input.dims.nbDims != 4) {
            spdlog::error("Invalid input dims");
            return false;
        }

        return true;
    }

    bool validate_recognizer(const std::string& path) {
        auto info = inspect_engine(path);
        info.print();

        if (!info.valid_engine) {
            spdlog::error("Invalid recognizer engine");
            return false;
        }

        bool has_input = false;
        bool has_output = false;

        for (const auto& tensor : info.inputs) {
            if (tensor.name == "input.1") {
                has_input = true;
            }
        }

        for (const auto& tensor : info.outputs) {
            if (tensor.name == "683") {
                has_output = true;
            }
        }

        if (!has_input || !has_output) {
            spdlog::error("Missing required tensors");
            return false;
        }

        return true;
    }

    bool validate_emotion(const std::string& path) {
        auto info = inspect_engine(path);
        info.print();

        if (!info.valid_engine) {
            spdlog::error("Invalid emotion engine");
            return false;
        }

        bool has_input = false;
        bool has_output = false;

        for (const auto& tensor : info.inputs) {
            if (tensor.name == "Input3") {
                has_input = true;
            }
        }

        for (const auto& tensor : info.outputs) {
            if (tensor.name == "Plus692_Output_0") {
                has_output = true;
            }
        }

        if (!has_input || !has_output) {
            spdlog::error("Missing required tensors");
            return false;
        }

        return true;
    }

    bool validate_age_gender(const std::string& path) {
        auto info = inspect_engine(path);
        info.print();

        if (!info.valid_engine) {
            spdlog::error("Invalid age/gender engine");
            return false;
        }

        bool has_input = false;
        bool has_output = false;

        for (const auto& tensor : info.inputs) {
            if (tensor.name == "pixel_values") {
                has_input = true;
            }
        }

        for (const auto& tensor : info.outputs) {
            if (tensor.name == "logits") {
                has_output = true;
            }
        }

        if (!has_input || !has_output) {
            spdlog::error("Missing required tensors");
            return false;
        }

        return true;
    }

    bool validate_all(const std::string& detector_path,
                      const std::string& recognizer_path,
                      bool enable_recognizer,
                      const std::string& emotion_path,
                      bool enable_emotion,
                      const std::string& age_gender_path,
                      bool enable_age_gender) {
        bool ok = true;

        ok &= validate_detector(detector_path);

        if (enable_recognizer) ok &= validate_recognizer(recognizer_path);
        if (enable_emotion) ok &= validate_emotion(emotion_path);
        if (enable_age_gender) ok &= validate_age_gender(age_gender_path);

        return ok;
    }
};
