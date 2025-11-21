// ============= include/model_validator.hpp =============
/*
 * Model Validator - Validación de Modelos TensorRT
 * 
 * FUNCIONALIDAD:
 * - Verifica existencia de archivos .engine
 * - Valida tensores de entrada/salida
 * - Muestra dimensiones y tipos de datos
 * - Detecta inconsistencias en configuración
 * 
 * USO:
 * ModelValidator validator;
 * if (!validator.validate_detector("models/scrfd.engine")) {
 *     exit(1);
 * }
 */

#pragma once
#include <string>
#include <vector>
#include <map>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include "tensorrt_logger.hpp"

struct TensorInfo {
    std::string name;
    nvinfer1::TensorIOMode mode;  // INPUT o OUTPUT
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
        spdlog::info("╔═══════════════════════════════════════╗");
        spdlog::info("║  Model: {:<30} ║", path.substr(path.find_last_of("/\\") + 1));
        spdlog::info("╠═══════════════════════════════════════╣");
        spdlog::info("║  File exists: {:<26} ║", exists ? "✓ YES" : "✗ NO");
        spdlog::info("║  Valid engine: {:<25} ║", valid_engine ? "✓ YES" : "✗ NO");
        spdlog::info("║  Size: {:<32} ║", std::to_string(total_size_mb) + " MB");
        spdlog::info("╠═══════════════════════════════════════╣");
        
        if (!inputs.empty()) {
            spdlog::info("║  INPUTS ({}):{:<26} ║", inputs.size(), "");
            for (const auto& tensor : inputs) {
                spdlog::info("║    • {:<34} ║", tensor.name);
                spdlog::info("║      Shape: {:<28} ║", tensor.dims_to_string());
                spdlog::info("║      Type: {:<29} ║", tensor.dtype_to_string());
            }
        }
        
        if (!outputs.empty()) {
            spdlog::info("╠═══════════════════════════════════════╣");
            spdlog::info("║  OUTPUTS ({}):{:<25} ║", outputs.size(), "");
            for (const auto& tensor : outputs) {
                spdlog::info("║    • {:<34} ║", tensor.name);
                spdlog::info("║      Shape: {:<28} ║", tensor.dims_to_string());
                spdlog::info("║      Type: {:<29} ║", tensor.dtype_to_string());
            }
        }
        
        spdlog::info("╚═══════════════════════════════════════╝");
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
        
        // Load engine
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
        
        // Inspect tensors
        int nb_tensors = engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; i++) {
            const char* name = engine->getIOTensorName(i);
            
            TensorInfo tensor;
            tensor.name = name;
            tensor.mode = engine->getTensorIOMode(name);
            tensor.dims = engine->getTensorShape(name);
            tensor.dtype = engine->getTensorDataType(name);
            
            // Calculate size
            size_t size = 1;
            for (int j = 0; j < tensor.dims.nbDims; j++) {
                size *= tensor.dims.d[j];
            }
            
            switch (tensor.dtype) {
                case nvinfer1::DataType::kFLOAT:
                case nvinfer1::DataType::kINT32:
                    tensor.size_bytes = size * 4;
                    break;
                case nvinfer1::DataType::kHALF:
                    tensor.size_bytes = size * 2;
                    break;
                case nvinfer1::DataType::kINT8:
                    tensor.size_bytes = size;
                    break;
                default:
                    tensor.size_bytes = size * 4;
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
    
    // Validar detector SCRFD
    bool validate_detector(const std::string& path) {
        spdlog::info("🔍 Validating SCRFD Detector...");
        auto info = inspect_engine(path);
        info.print();
        
        if (!info.valid_engine) {
            spdlog::error("❌ Invalid detector engine");
            return false;
        }
        
        // Validaciones específicas
        if (info.inputs.size() != 1) {
            spdlog::error("❌ Expected 1 input, got {}", info.inputs.size());
            return false;
        }
        
        if (info.outputs.size() != 9) {
            spdlog::warn("⚠️ Expected 9 outputs for SCRFD, got {}", info.outputs.size());
        }
        
        // Verificar input shape (debería ser [1, 3, 640, 640])
        auto& input = info.inputs[0];
        if (input.dims.nbDims != 4) {
            spdlog::error("❌ Invalid input dimensions");
            return false;
        }
        
        spdlog::info("✓ Detector validation passed");
        return true;
    }
    
    // Validar ArcFace
    bool validate_recognizer(const std::string& path) {
        spdlog::info("🔍 Validating ArcFace Recognizer...");
        auto info = inspect_engine(path);
        info.print();
        
        if (!info.valid_engine) {
            spdlog::error("❌ Invalid recognizer engine");
            return false;
        }
        
        // Verificar tensores esperados
        bool has_input = false;
        bool has_output = false;
        
        for (const auto& tensor : info.inputs) {
            if (tensor.name == "input.1") {
                has_input = true;
                // Verificar shape [1, 3, 112, 112]
                if (tensor.dims.nbDims == 4 &&
                    tensor.dims.d[0] == 1 &&
                    tensor.dims.d[1] == 3 &&
                    tensor.dims.d[2] == 112 &&
                    tensor.dims.d[3] == 112) {
                    spdlog::info("✓ Input tensor 'input.1' is valid");
                } else {
                    spdlog::error("❌ Invalid input.1 shape: {}", tensor.dims_to_string());
                    return false;
                }
            }
        }
        
        for (const auto& tensor : info.outputs) {
            if (tensor.name == "683") {
                has_output = true;
                // Verificar shape [1, 512]
                if (tensor.dims.nbDims == 2 &&
                    tensor.dims.d[0] == 1 &&
                    tensor.dims.d[1] == 512) {
                    spdlog::info("✓ Output tensor '683' is valid");
                } else {
                    spdlog::error("❌ Invalid 683 shape: {}", tensor.dims_to_string());
                    return false;
                }
            }
        }
        
        if (!has_input || !has_output) {
            spdlog::error("❌ Missing expected tensors (input.1, 683)");
            return false;
        }
        
        spdlog::info("✓ Recognizer validation passed");
        return true;
    }
    
    // Validar Emotion
    bool validate_emotion(const std::string& path) {
        spdlog::info("🔍 Validating Emotion Recognizer...");
        auto info = inspect_engine(path);
        info.print();
        
        if (!info.valid_engine) {
            spdlog::error("❌ Invalid emotion engine");
            return false;
        }
        
        // FER+ debe tener input "Input3" y output "Plus692_Output_0"
        bool has_input = false;
        bool has_output = false;
        
        for (const auto& tensor : info.inputs) {
            if (tensor.name == "Input3") {
                has_input = true;
                spdlog::info("✓ Found input tensor 'Input3'");
            }
        }
        
        for (const auto& tensor : info.outputs) {
            if (tensor.name == "Plus692_Output_0") {
                has_output = true;
                spdlog::info("✓ Found output tensor 'Plus692_Output_0'");
            }
        }
        
        if (!has_input || !has_output) {
            spdlog::error("❌ Missing expected tensors (Input3, Plus692_Output_0)");
            spdlog::error("Available tensors:");
            for (const auto& t : info.inputs) {
                spdlog::error("  Input: {}", t.name);
            }
            for (const auto& t : info.outputs) {
                spdlog::error("  Output: {}", t.name);
            }
            return false;
        }
        
        spdlog::info("✓ Emotion validation passed");
        return true;
    }
    
    // Validar Age/Gender
    bool validate_age_gender(const std::string& path) {
        spdlog::info("🔍 Validating Age/Gender Predictor...");
        auto info = inspect_engine(path);
        info.print();
        
        if (!info.valid_engine) {
            spdlog::error("❌ Invalid age/gender engine");
            return false;
        }
        
        // Debe tener input "pixel_values" y output "logits"
        bool has_input = false;
        bool has_output = false;
        
        for (const auto& tensor : info.inputs) {
            if (tensor.name == "pixel_values") {
                has_input = true;
                spdlog::info("✓ Found input tensor 'pixel_values'");
            }
        }
        
        for (const auto& tensor : info.outputs) {
            if (tensor.name == "logits") {
                has_output = true;
                spdlog::info("✓ Found output tensor 'logits'");
            }
        }
        
        if (!has_input || !has_output) {
            spdlog::error("❌ Missing expected tensors (pixel_values, logits)");
            spdlog::error("Available tensors:");
            for (const auto& t : info.inputs) {
                spdlog::error("  Input: {}", t.name);
            }
            for (const auto& t : info.outputs) {
                spdlog::error("  Output: {}", t.name);
            }
            return false;
        }
        
        spdlog::info("✓ Age/Gender validation passed");
        return true;
    }
    
    // Validar todos los modelos (con opcionales)
    bool validate_all(const std::string& detector_path,
                     const std::string& recognizer_path,
                     bool enable_recognizer,
                     const std::string& emotion_path,
                     bool enable_emotion,
                     const std::string& age_gender_path,
                     bool enable_age_gender) {
        bool all_valid = true;
        
        spdlog::info("╔════════════════════════════════════════╗");
        spdlog::info("║    VALIDATING ALL MODELS               ║");
        spdlog::info("╚════════════════════════════════════════╝");
        
        // Detector (siempre requerido)
        all_valid &= validate_detector(detector_path);
        
        // Recognizer (solo si está habilitado)
        if (enable_recognizer) {
            all_valid &= validate_recognizer(recognizer_path);
        } else {
            spdlog::info("⊘ Recognizer disabled - skipping validation");
        }
        
        // Emotion (solo si está habilitado)
        if (enable_emotion) {
            all_valid &= validate_emotion(emotion_path);
        } else {
            spdlog::info("⊘ Emotion disabled - skipping validation");
        }
        
        // Age/Gender (solo si está habilitado)
        if (enable_age_gender) {
            all_valid &= validate_age_gender(age_gender_path);
        } else {
            spdlog::info("⊘ Age/Gender disabled - skipping validation");
        }
        
        if (all_valid) {
            spdlog::info("╔════════════════════════════════════════╗");
            spdlog::info("║  ✓ ALL MODELS VALIDATED SUCCESSFULLY   ║");
            spdlog::info("╚════════════════════════════════════════╝");
        } else {
            spdlog::error("╔════════════════════════════════════════╗");
            spdlog::error("║  ✗ MODEL VALIDATION FAILED             ║");
            spdlog::error("╚════════════════════════════════════════╝");
        }
        
        return all_valid;
    }
};