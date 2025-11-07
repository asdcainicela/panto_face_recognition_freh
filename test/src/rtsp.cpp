#include "stream_viewer.hpp"
#include <thread>
#include <chrono>

int main() {
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";        
    int port = 554;

    // Crear viewers
    StreamViewer viewer_main(user, pass, ip, port, "main");
    StreamViewer viewer_sub(user, pass, ip, port, "sub");

    // Ejecutar en hilos separados con delay
    std::thread thread_main([&]() { viewer_main.run(); });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Delay entre ventanas
    
    std::thread thread_sub([&]() { viewer_sub.run(); });

    // Esperar a que ambos terminen
    thread_main.join();
    thread_sub.join();

    return 0;
}