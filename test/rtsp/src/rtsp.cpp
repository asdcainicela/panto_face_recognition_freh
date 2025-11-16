#include "stream_viewer.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <csignal>
#include <atomic>
#include <iostream>
#include <future>

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

// Función para inicializar un viewer en paralelo
StreamViewer* init_viewer_async(const std::string& user, const std::string& pass,
                                const std::string& ip, int port, const std::string& stream_type,
                                bool enable_recording, const std::string& output_dir,
                                const std::string& record_stream, int duration,
                                std::atomic<bool>* stop_sig) {
    try {
        spdlog::info("⚡ inicializando stream {} en paralelo...", stream_type);
        
        StreamViewer* viewer = new StreamViewer(user, pass, ip, port, stream_type);
        viewer->set_stop_signal(stop_sig);
        
        if (enable_recording && (record_stream == stream_type || record_stream == "both")) {
            viewer->enable_recording(output_dir);
        }
        
        if (duration > 0) {
            viewer->set_max_duration(duration);
        }
        
        spdlog::info("✓ viewer {} inicializado", stream_type);
        return viewer;
        
    } catch (const std::exception& e) {
        spdlog::error("✗ error inicializando viewer {}: {}", stream_type, e.what());
        return nullptr;
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
    
    spdlog::info("🚀 iniciando aplicación RTSP...");
    
    StreamViewer* viewer_main = nullptr;
    StreamViewer* viewer_sub = nullptr;
    
    // ============================================
    // INICIALIZACIÓN PARALELA DE AMBOS STREAMS
    // ============================================
    if (stream_type == "both") {
        spdlog::info("⚡ inicializando AMBOS streams en paralelo...");
        
        // Crear futures para inicialización paralela
        auto future_main = std::async(std::launch::async, init_viewer_async,
            user, pass, ip, port, "main", 
            enable_recording, output_dir, record_stream, duration, &stop_signal);
        
        auto future_sub = std::async(std::launch::async, init_viewer_async,
            user, pass, ip, port, "sub",
            enable_recording, output_dir, record_stream, duration, &stop_signal);
        
        // Esperar a que ambos terminen (se ejecutan en paralelo)
        viewer_main = future_main.get();
        viewer_sub = future_sub.get();
        
        if (viewer_main && viewer_sub) {
            spdlog::info("✓✓ ambos streams inicializados exitosamente");
        } else {
            spdlog::warn("⚠️ algunos streams no se inicializaron correctamente");
        }
    }
    // Inicialización secuencial si solo se pide uno
    else if (stream_type == "main") {
        viewer_main = init_viewer_async(user, pass, ip, port, "main",
            enable_recording, output_dir, record_stream, duration, &stop_signal);
    }
    else if (stream_type == "sub") {
        viewer_sub = init_viewer_async(user, pass, ip, port, "sub",
            enable_recording, output_dir, record_stream, duration, &stop_signal);
    }
    
    // ============================================
    // EJECUTAR STREAMS EN THREADS SEPARADOS
    // ============================================
    std::thread thread_main, thread_sub;
    
    if (viewer_main) {
        spdlog::info("▶️ lanzando thread para stream MAIN");
        thread_main = std::thread([&]() { viewer_main->run(); });
    }
    
    if (viewer_sub) {
        spdlog::info("▶️ lanzando thread para stream SUB");
        thread_sub = std::thread([&]() { viewer_sub->run(); });
    }
    
    // Esperar a que terminen
    if (thread_main.joinable()) {
        spdlog::debug("esperando thread MAIN...");
        thread_main.join();
    }
    
    if (thread_sub.joinable()) {
        spdlog::debug("esperando thread SUB...");
        thread_sub.join();
    }
    
    // Limpieza
    delete viewer_main;
    delete viewer_sub;

    spdlog::info("🏁 aplicación finalizada correctamente");
    return 0;
}