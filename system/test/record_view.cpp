
// ============= test/record_view.cpp =============
#include "stream_capture.hpp"
#include "config.hpp"
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
    
    std::string stream_type = argc >= 2 ? argv[1] : "main";
    int duration = argc >= 3 ? std::atoi(argv[2]) : 0;
    
    spdlog::info("grabando y visualizando stream: {}", stream_type);
    
    StreamCapture capture(Config::DEFAULT_USER, Config::DEFAULT_PASS, 
                         Config::DEFAULT_IP, Config::DEFAULT_PORT, stream_type);
    capture.enable_recording("../videos");
    capture.enable_viewing(cv::Size(640, 480));
    capture.set_stop_signal(&stop_signal);
    if (duration > 0) capture.set_max_duration(duration);
    
    capture.run();
    
    spdlog::info("completado");
    return 0;
}