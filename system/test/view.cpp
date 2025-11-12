
// ============= test/view.cpp =============
#include "stream_capture.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("cerrando visualizacion...");
        stop_signal = true;
    }
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::vector<std::string> streams;
    
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "main" || arg == "sub") {
            streams.push_back(arg);
        } else {
            streams = {"main", "sub"};
        }
    } else {
        streams = {"main", "sub"};
    }
    
    spdlog::info("visualizando streams...");
    
    std::vector<StreamCapture*> captures;
    std::vector<std::thread> threads;
    
    for (const auto& s : streams) {
        auto* cap = new StreamCapture(Config::DEFAULT_USER, Config::DEFAULT_PASS, 
                                      Config::DEFAULT_IP, Config::DEFAULT_PORT, s);
        cap->enable_viewing(cv::Size(Config::DEFAULT_DISPLAY_WIDTH, Config::DEFAULT_DISPLAY_HEIGHT));
        cap->set_stop_signal(&stop_signal);
        captures.push_back(cap);
        
        threads.emplace_back([cap]() { cap->run(); });
    }
    
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    for (auto* cap : captures) {
        delete cap;
    }
    
    spdlog::info("visualizacion finalizada");
    return 0;
}
