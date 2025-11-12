#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("deteniendo...");
        stop_signal = true;
    }
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";
    int port = 554;
    
    std::string stream_type = "main";
    int duration = 0;
    
    if (argc >= 2) {
        stream_type = argv[1];
    }
    if (argc >= 3) {
        duration = std::atoi(argv[2]);
    }
    
    spdlog::info("grabando y visualizando stream: {}", stream_type);
    if (duration > 0) {
        spdlog::info("duracion: {} segundos", duration);
    } else {
        spdlog::info("presiona Ctrl+C o ESC para detener");
    }
    
    StreamCapture capture(user, pass, ip, port, stream_type);
    capture.enable_recording("../videos");
    capture.set_stop_signal(&stop_signal);
    if (duration > 0) {
        capture.set_max_duration(duration);
    }
    
    if (!capture.open()) {
        spdlog::error("error al abrir stream");
        return 1;
    }
    
    std::string window_name = "record + view - " + stream_type;
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 640, 480);
    
    cv::Mat frame, display;
    while (capture.read(frame)) {
        cv::resize(frame, display, cv::Size(640, 480));
        
        const auto& stats = capture.get_stats();
        
        DrawUtils::DrawConfig config;
        config.show_recording = capture.is_recording();
        if (capture.is_recording()) {
            config.color = cv::Scalar(0, 0, 255);
        }
        
        DrawUtils::draw_stream_info(display, stats, stream_type, config);
        DrawUtils::draw_recording_indicator(display, capture.is_recording(), config);
        
        cv::imshow(window_name, display);
        
        if (cv::waitKey(1) == 27) {
            stop_signal = true;
            break;
        }
        
        capture.print_stats();
    }
    
    cv::destroyWindow(window_name);
    capture.print_final_stats();
    spdlog::info("completado");
    
    return 0;
}