#include "stream_viewer.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

int main() {
    
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v"); // Configurar spdlog
    spdlog::set_level(spdlog::level::info);
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";        
    int port = 554;

    spdlog::info("iniciando stream viewers...");

    StreamViewer viewer_main(user, pass, ip, port, "main", cv::Size(640, 480));
    StreamViewer viewer_sub(user, pass, ip, port, "sub", cv::Size(640, 480));

    // Ejecutar en hilos separados con delay
    std::thread thread_main([&]() { viewer_main.run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::thread thread_sub([&]() { viewer_sub.run(); });

    // Esperar a que ambos terminen
    thread_main.join();
    thread_sub.join();

    spdlog::info("finalizando programa");
    return 0;
}