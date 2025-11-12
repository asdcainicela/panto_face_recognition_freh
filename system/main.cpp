#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <fstream>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("deteniendo sistema...");
        stop_signal = true;
    }
}

struct Config {
    std::string stream_type = "main";
    bool show_view = true;
    bool show_info = true;
    bool enable_record = false;
    int duration = 0;
    std::string profile = "config_1080p_roi.toml";
};

Config load_config() {
    Config cfg;
    
    // TODO: Parsear config.toml aqui
    // Por ahora valores por defecto
    
    return cfg;
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
    
    Config cfg = load_config();
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            cfg.profile = argv[++i];
        }
    }
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";
    int port = 554;
    
    spdlog::info("iniciando panto...");
    spdlog::info("perfil: {}", cfg.profile);
    
    StreamCapture capture(user, pass, ip, port, cfg.stream_type);
    capture.set_stop_signal(&stop_signal);
    
    if (cfg.enable_record) {
        capture.enable_recording("../videos");
    }
    
    if (cfg.duration > 0) {
        capture.set_max_duration(cfg.duration);
    }
    
    if (!capture.open()) {
        spdlog::error("error al abrir stream");
        return 1;
    }
    
    // Detectar resolucion y sugerir config
    cv::Mat first_frame;
    if (capture.read(first_frame)) {
        const auto& stats = capture.get_stats();
        std::string suggested = detect_resolution_config(stats.resolution);
        spdlog::info("resolucion detectada: {}x{}", stats.resolution.width, stats.resolution.height);
        spdlog::info("config sugerido: {}", suggested);
    }
    
    if (cfg.show_view) {
        std::string window_name = "panto - " + cfg.stream_type;
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 640, 480);
        
        cv::Mat frame, display;
        while (capture.read(frame)) {
            cv::resize(frame, display, cv::Size(640, 480));
            
            if (cfg.show_info) {
                const auto& stats = capture.get_stats();
                DrawUtils::DrawConfig config;
                DrawUtils::draw_stream_info(display, stats, cfg.stream_type, config);
            }
            
            cv::imshow(window_name, display);
            
            if (cv::waitKey(1) == 27) break;
            
            capture.print_stats();
        }
        
        cv::destroyWindow(window_name);
    } else {
        cv::Mat frame;
        while (capture.read(frame)) {
            capture.print_stats();
        }
    }
    
    capture.print_final_stats();
    spdlog::info("panto finalizado");
    
    return 0;
}