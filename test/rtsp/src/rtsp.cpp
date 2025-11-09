#include "stream_viewer.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <csignal>
#include <atomic>
#include <iostream>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("Ctrl+C recibido, cerrando...");
        stop_signal = true;
    }
}

void print_usage() {
    std::cout << "\nUso: ./rtsp [opciones]\n\n"
              << "  -s, --stream <tipo>     main, sub, both (default: both)\n"
              << "  -r, --record            Habilitar grabación\n"
              << "  --record-stream <tipo>  Grabar: main, sub, both\n"
              << "  -d, --duration <seg>    Duración en segundos\n"
              << "  --output-dir <dir>      Directorio salida\n"
              << "  -h, --help              Ayuda\n\n"
              << "Ejemplos:\n"
              << "  ./rtsp\n"
              << "  ./rtsp -r\n"
              << "  ./rtsp -s main -r -d 60\n"
              << std::endl;
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
    
    std::string stream_type = "both";
    std::string record_stream = "";
    bool enable_recording = false;
    int duration = 0;
    std::string output_dir = "../videos_rtsp";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
        else if ((arg == "-s" || arg == "--stream") && i + 1 < argc) stream_type = argv[++i];
        else if (arg == "-r" || arg == "--record") enable_recording = true;
        else if (arg == "--record-stream" && i + 1 < argc) record_stream = argv[++i];
        else if ((arg == "-d" || arg == "--duration") && i + 1 < argc) duration = std::atoi(argv[++i]);
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
    }
    
    if (record_stream.empty()) record_stream = stream_type;
    
    spdlog::info("iniciando...");
    
    StreamViewer* viewer_main = nullptr;
    StreamViewer* viewer_sub = nullptr;
    
    if (stream_type == "main" || stream_type == "both") {
        viewer_main = new StreamViewer(user, pass, ip, port, "main");
        viewer_main->set_stop_signal(&stop_signal);
        if (enable_recording && (record_stream == "main" || record_stream == "both")) {
            viewer_main->enable_recording(output_dir);
        }
        if (duration > 0) viewer_main->set_max_duration(duration);
    }
    
    if (stream_type == "sub" || stream_type == "both") {
        viewer_sub = new StreamViewer(user, pass, ip, port, "sub");
        viewer_sub->set_stop_signal(&stop_signal);
        if (enable_recording && (record_stream == "sub" || record_stream == "both")) {
            viewer_sub->enable_recording(output_dir);
        }
        if (duration > 0) viewer_sub->set_max_duration(duration);
    }
    
    std::thread thread_main, thread_sub;
    
    if (viewer_main) {
        thread_main = std::thread([&]() { viewer_main->run(); });
    }
    
    if (viewer_sub) {
        thread_sub = std::thread([&]() { viewer_sub->run(); });
    }
    
    if (thread_main.joinable()) thread_main.join();
    if (thread_sub.joinable()) thread_sub.join();
    
    delete viewer_main;
    delete viewer_sub;

    spdlog::info("finalizado");
    return 0;
}