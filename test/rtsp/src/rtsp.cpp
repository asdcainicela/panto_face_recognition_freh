#include "stream_viewer.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <iostream>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("señal recibida (Ctrl+C), deteniendo streams...");
        stop_signal = true;
    }
}

void print_usage() {
    std::cout << "\nUso: ./rtsp [opciones]\n\n"
              << "Opciones:\n"
              << "  --stream, -s <tipo>     Stream: main, sub, both (default: both)\n"
              << "  --record, -r            Habilitar grabación\n"
              << "  --record-stream <tipo>  Grabar: main, sub, both (default: igual a --stream)\n"
              << "  --duration, -d <seg>    Duración en segundos (0 = hasta Ctrl+C)\n"
              << "  --output-dir <dir>      Directorio salida (default: ../videos_rtsp)\n"
              << "  --help, -h              Mostrar ayuda\n\n"
              << "Ejemplos:\n"
              << "  ./rtsp                  # Ver ambos streams\n"
              << "  ./rtsp -r               # Grabar ambos\n"
              << "  ./rtsp -s main -r -d 60 # Grabar main por 60s\n"
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
        
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        else if (arg == "--stream" || arg == "-s") {
            if (i + 1 < argc) stream_type = argv[++i];
        }
        else if (arg == "--record" || arg == "-r") {
            enable_recording = true;
        }
        else if (arg == "--record-stream") {
            if (i + 1 < argc) record_stream = argv[++i];
        }
        else if (arg == "--duration" || arg == "-d") {
            if (i + 1 < argc) duration = std::atoi(argv[++i]);
        }
        else if (arg == "--output-dir") {
            if (i + 1 < argc) output_dir = argv[++i];
        }
        else {
            spdlog::warn("argumento desconocido: {}", arg);
        }
    }
    
    if (record_stream.empty()) {
        record_stream = stream_type;
    }
    
    if (stream_type != "main" && stream_type != "sub" && stream_type != "both") {
        spdlog::error("stream inválido: {}. Use: main, sub o both", stream_type);
        return -1;
    }
    
    if (enable_recording && record_stream != "main" && record_stream != "sub" && record_stream != "both") {
        spdlog::error("record-stream inválido: {}. Use: main, sub o both", record_stream);
        return -1;
    }
    
    spdlog::info("iniciando stream viewers...");
    spdlog::info("visualización: {}", stream_type);
    if (enable_recording) {
        spdlog::info("grabación habilitada para: {}", record_stream);
        spdlog::info("directorio de salida: {}", output_dir);
        if (duration > 0) {
            spdlog::info("duración: {} segundos", duration);
        } else {
            spdlog::info("presiona Ctrl+C para detener");
        }
    }
    
    StreamViewer* viewer_main = nullptr;
    StreamViewer* viewer_sub = nullptr;
    
    if (stream_type == "main" || stream_type == "both") {
        viewer_main = new StreamViewer(user, pass, ip, port, "main", cv::Size(640, 480));
        viewer_main->set_stop_signal(&stop_signal);
        
        if (enable_recording && (record_stream == "main" || record_stream == "both")) {
            viewer_main->enable_recording(output_dir);
        }
        
        if (duration > 0) {
            viewer_main->set_max_duration(duration);
        }
    }
    
    if (stream_type == "sub" || stream_type == "both") {
        viewer_sub = new StreamViewer(user, pass, ip, port, "sub", cv::Size(640, 480));
        viewer_sub->set_stop_signal(&stop_signal);
        
        if (enable_recording && (record_stream == "sub" || record_stream == "both")) {
            viewer_sub->enable_recording(output_dir);
        }
        
        if (duration > 0) {
            viewer_sub->set_max_duration(duration);
        }
    }
    
    std::thread* thread_main = nullptr;
    std::thread* thread_sub = nullptr;
    
    if (viewer_main) {
        thread_main = new std::thread([&]() { viewer_main->run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    if (viewer_sub) {
        thread_sub = new std::thread([&]() { viewer_sub->run(); });
    }
    
    if (thread_main) thread_main->join();
    if (thread_sub) thread_sub->join();
    
    delete thread_main;
    delete thread_sub;
    delete viewer_main;
    delete viewer_sub;

    spdlog::info("programa finalizado");
    return 0;
}