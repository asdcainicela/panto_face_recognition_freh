#include "stream_viewer.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

int main() {
    
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // Configurar spdlog a nivel info
    spdlog::set_level(spdlog::level::info);
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";        
    auto port = 554;

    spdlog::info("iniciando stream viewers...");

    StreamViewer viewer_main(user, pass, ip, port, "main", cv::Size(640, 480));
    StreamViewer viewer_sub(user, pass, ip, port, "sub", cv::Size(640, 480));


    std::thread thread_main([&]() { viewer_main.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::thread thread_sub([&]() { viewer_sub.run(); });

    
    thread_main.join();
    thread_sub.join();

    spdlog::info("programa finalizado");
    return 0;
}