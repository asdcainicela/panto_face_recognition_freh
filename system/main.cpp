#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <fstream>
#include <map>
#include <stdexcept>

// Simple TOML parser (sin librería externa)
class SimpleToml {
private:
    std::map<std::string, std::string> values;
    
    std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t");
        if (start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t");
        return s.substr(start, end - start + 1);
    }
    
public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            spdlog::warn("no se pudo abrir config: {}", filename);
            return false;
        }
        
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
                
                // Remover comillas
                if (val.front() == '"' && val.back() == '"') {
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
    
    int get_int(const std::string& key, int def = 0) const {
        try {
            return std::stoi(get(key));
        } catch (...) {
            return def;
        }
    }
    
    bool get_bool(const std::string& key, bool def = false) const {
        std::string val = get(key);
        return val == "true" || val == "1";
    }
    
    void dump() const {
        spdlog::debug("=== Config Loaded ===");
        for (const auto& kv : values) {
            spdlog::debug("  {}: {}", kv.first, kv.second);
        }
    }
};

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("deteniendo sistema...");
        stop_signal = true;
    }
}

std::string detect_resolution_config(const cv::Size& resolution) {
    int pixels = resolution.width * resolution.height;
    
    if (pixels >= 3840 * 2160 * 0.9) {
        return "configs/config_4k.toml";
    } else if (pixels >= 2560 * 1440 * 0.9) {
        return "configs/config_1440p.toml";
    } else if (pixels >= 1920 * 1080 * 0.9) {
        return "configs/config_1080p_roi.toml";
    } else if (pixels >= 1280 * 720 * 0.9) {
        return "configs/config_720p.toml";
    }
    
    return "configs/config_1080p_roi.toml";
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse argumentos
    std::string config_file = "configs/config_1080p_roi.toml";
    bool verbose = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
            spdlog::set_level(spdlog::level::debug);
        }
    }
    
    spdlog::info("=== PANTO Sistema de Reconocimiento ===");
    spdlog::info("config: {}", config_file);
    
    // Cargar configuración
    SimpleToml config;
    if (!config.load(config_file)) {
        spdlog::warn("usando valores por defecto");
    }
    
    if (verbose) {
        config.dump();
    }
    
    // Leer valores de config
    std::string stream_type = "main";
    bool enable_display = config.get_bool("output.display_output", true);
    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_detections = config.get_bool("output.draw_detections", true);
    bool draw_roi = config.get_bool("output.draw_roi", true);
    
    // Credenciales (aún desde config.hpp, pero podrían venir de TOML)
    std::string user = Config::DEFAULT_USER;
    std::string pass = Config::DEFAULT_PASS;
    std::string ip = Config::DEFAULT_IP;
    int port = Config::DEFAULT_PORT;
    
    spdlog::info("conectando a rtsp://{}@{}:{}/{}", user, ip, port, stream_type);
    
    // Crear captura
    StreamCapture capture(user, pass, ip, port, stream_type);
    capture.set_stop_signal(&stop_signal);
    
    if (!capture.open()) {
        spdlog::error("error al abrir stream");
        return 1;
    }
    
    // Detectar resolución y sugerir config óptimo
    cv::Mat first_frame;
    if (capture.read(first_frame)) {
        const auto& stats = capture.get_stats();
        std::string suggested = detect_resolution_config(stats.resolution);
        
        spdlog::info("resolucion detectada: {}x{}", 
                     stats.resolution.width, stats.resolution.height);
        
        if (suggested != config_file) {
            spdlog::info("config sugerido: {}", suggested);
        }
    }
    
    // Loop principal
    if (enable_display) {
        std::string window_name = "PANTO - " + stream_type;
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, Config::DEFAULT_DISPLAY_WIDTH, 
                        Config::DEFAULT_DISPLAY_HEIGHT);
        
        cv::Mat frame, display;
        while (capture.read(frame)) {
            cv::resize(frame, display, cv::Size(Config::DEFAULT_DISPLAY_WIDTH, 
                                               Config::DEFAULT_DISPLAY_HEIGHT));
            
            const auto& stats = capture.get_stats();
            DrawUtils::DrawConfig draw_config;
            draw_config.show_fps = draw_fps;
            
            DrawUtils::draw_stream_info(display, stats, stream_type, draw_config);
            
            cv::imshow(window_name, display);
            
            if (cv::waitKey(1) == 27) break;
            
            if (stats.frames % Config::DEFAULT_FPS_INTERVAL == 0) {
                capture.print_stats();
            }
        }
        
        cv::destroyWindow(window_name);
    } else {
        // Headless mode
        cv::Mat frame;
        while (capture.read(frame)) {
            capture.print_stats();
        }
    }
    
    capture.print_final_stats();
    spdlog::info("panto finalizado");
    
    return 0;
}