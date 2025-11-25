// ============= tests/test_validator.cpp =============
//
// Test unitario para validar modelos TensorRT
//
// USO:
//   ./test_validator [config.toml]
//
// EJEMPLO:
//   ./test_validator config.toml
//

#include "model_validator.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <map>

// ==================== SIMPLE TOML PARSER ====================
class SimpleToml {
private:
    std::map<std::string, std::string> values;

    std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string line, section;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            if (line[0] == '[' && line.back() == ']') {
                section = line.substr(1, line.length() - 2);
                continue;
            }

            auto eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = trim(line.substr(0, eq));
                std::string val = trim(line.substr(eq + 1));

                if (!val.empty() && val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.length() - 2);
                }

                std::string full_key = section.empty() ? key : section + "." + key;
                values[full_key] = val;
            }
        }
        return true;
    }

    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = values.find(key);
        return it != values.end() ? it->second : def;
    }

    bool get_bool(const std::string& key, bool def = false) const {
        std::string val = get(key);
        return val == "true" || val == "1";
    }
};

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S] [%l] %v");
    spdlog::set_level(spdlog::level::info);

    std::string config_file = argc >= 2 ? argv[1] : "config.toml";

    spdlog::info("Model Validator Test");
    spdlog::info("Configuracion: {}", config_file);

    // Load config
    SimpleToml config;
    if (!config.load(config_file)) {
        spdlog::error("No se pudo cargar {}", config_file);
        return 1;
    }

    // Get model paths
    std::string detector_path     = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    std::string recognizer_path   = config.get("recognizer.model_path", "models/arcface_r100.engine");
    std::string emotion_path      = config.get("emotion.model_path", "models/emotion_ferplus.engine");
    std::string age_gender_path   = config.get("age_gender.model_path", "models/age_gender.engine");

    // Get enabled flags
    bool mode_recognize     = config.get_bool("mode.recognize", true);
    bool emotion_enabled    = config.get_bool("emotion.enabled", false);
    bool age_gender_enabled = config.get_bool("age_gender.enabled", false);

    spdlog::info("Detector: {}", detector_path);
    spdlog::info("Recognizer: {} ({})", recognizer_path, mode_recognize ? "enabled" : "disabled");
    spdlog::info("Emotion: {} ({})", emotion_path, emotion_enabled ? "enabled" : "disabled");
    spdlog::info("Age/Gender: {} ({})", age_gender_path, age_gender_enabled ? "enabled" : "disabled");

    // Validate models
    ModelValidator validator;

    bool success = validator.validate_all(
        detector_path,
        recognizer_path,
        mode_recognize,
        emotion_path,
        emotion_enabled,
        age_gender_path,
        age_gender_enabled
    );

    if (success) {
        spdlog::info("Test completado. Modelos validados correctamente.");
        return 0;
    } else {
        spdlog::error("Test fallido. Hay problemas con uno o mas modelos.");
        return 1;
    }
}
