// ============= test/record.cpp =============
#include "stream_capture.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("deteniendo grabacion...");
        stop_signal = true;
    }
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::string stream_type = argc >= 2 ? argv[1] : Config::DEFAULT_STREAM;
    int duration = argc >= 3 ? std::atoi(argv[2]) : 0;
    
    spdlog::info("iniciando grabacion stream: {}", stream_type);
    
    StreamCapture capture(Config::DEFAULT_USER, Config::DEFAULT_PASS, 
                         Config::DEFAULT_IP, Config::DEFAULT_PORT, stream_type);
    capture.enable_recording(Config::DEFAULT_OUTPUT_DIR);
    capture.set_stop_signal(&stop_signal);
    if (duration > 0) capture.set_max_duration(duration);
    
    capture.run();
    
    spdlog::info("grabacion completada");
    return 0;
}
