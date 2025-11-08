/*
# Ir al directorio build
cd /workspace/panto_face_recognition_freh/test/build

# ==========================================
# COMANDOS BÁSICOS
# ==========================================

# Grabar stream "main" hasta presionar Ctrl+C
./record_video

# Grabar por 60 segundos
./record_video 60

# Grabar por 120 segundos
./record_video 120

# ==========================================
# CON PREVIEW (ventana visual)
# ==========================================

# Grabar con preview hasta Ctrl+C
./record_video --preview

# Grabar con preview por 30 segundos
./record_video --preview 30

# También puedes usar -p
./record_video -p

# ==========================================
# CAMBIAR STREAM (main o sub)
# ==========================================

# Grabar stream "sub"
./record_video --stream sub

# Grabar stream "sub" por 60 segundos
./record_video --stream sub 60

# Grabar stream "sub" con preview
./record_video --stream sub --preview

# También puedes usar -s
./record_video -s sub

# ==========================================
# COMBINACIONES AVANZADAS
# ==========================================

# Stream "sub" con preview por 30 segundos
./record_video --stream sub --preview --duration 30

# Forma corta (equivalente)
./record_video -s sub -p -d 30

# Stream "main" por 120 segundos sin preview
./record_video --stream main --duration 120

# Forma corta
./record_video -s main -d 120

# ==========================================
# VER VIDEOS GRABADOS
# ==========================================

# Listar videos grabados
ls -lh ../videos_rtsp/

# Ver el último video grabado
ls -lt ../videos_rtsp/ | head -n 2

# Desde tu host (fuera del Docker)
ls ~/jetson_workspace/panto_face_recognition_freh/videos_rtsp/
*/
#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <csignal>
#include <atomic>
#include <thread>
#include <fstream>

// Variable global para manejar la señal
std::atomic<bool> stop_recording(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("señal recibida, deteniendo grabación...");
        stop_recording = true;
    }
}

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    // Registrar manejador de señales
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configuración
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";
    int port = 554;
    std::string stream_type = "main"; // o "sub"
    
    // Duración de grabación en segundos (0 = hasta Ctrl+C)
    int duration_seconds = 0;
    bool show_preview = false;
    
    // Parsear argumentos
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--preview" || arg == "-p") {
            show_preview = true;
        } else if (arg == "--stream" || arg == "-s") {
            if (i + 1 < argc) {
                stream_type = argv[++i];
            }
        } else if (arg == "--duration" || arg == "-d") {
            if (i + 1 < argc) {
                duration_seconds = std::atoi(argv[++i]);
            }
        } else {
            duration_seconds = std::atoi(arg.c_str());
        }
    }
    
    spdlog::info("iniciando grabación de stream {}...", stream_type);
    
    // Construir pipeline
    std::string pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    
    // Abrir stream
    cv::VideoCapture cap;
    try {
        cap = open_cap(pipeline);
    } catch (const std::exception& e) {
        spdlog::error("error al conectar: {}", e.what());
        return -1;
    }
    
    // Obtener propiedades del video
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    if (fps <= 0) fps = 25.0; // FPS por defecto
    
    spdlog::info("resolución: {}x{} @ {} fps", frame_width, frame_height, fps);
    
    // Crear archivo de salida (en /workspace si existe, sino en directorio actual)
    std::string output_dir = "/workspace/videos";
    if (std::ifstream("/workspace").good()) {
        // Crear carpeta videos si no existe
        system("mkdir -p /workspace/videos");
    } else {
        output_dir = ".";
    }
    
    std::string filename = output_dir + "/recording_" + stream_type + "_" + get_timestamp() + ".mp4";
    cv::VideoWriter writer;
    
    // Codec H.264
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(filename, fourcc, fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        spdlog::error("no se pudo crear el archivo de video: {}", filename);
        cap.release();
        return -1;
    }
    
    spdlog::info("grabando en: {}", filename);
    if (duration_seconds > 0) {
        spdlog::info("duración: {} segundos", duration_seconds);
    } else {
        spdlog::info("presiona Ctrl+C para detener");
    }
    
    // Variables de control
    cv::Mat frame;
    int frames_recorded = 0;
    int frames_lost = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_log = start_time;
    
    // Ventana opcional para preview
    if (show_preview) {
        cv::namedWindow("Recording", cv::WINDOW_NORMAL);
        cv::resizeWindow("Recording", 640, 480);
    }
    
    while (!stop_recording) {
        if (!cap.read(frame)) {
            frames_lost++;
            spdlog::warn("frame perdido. reconectando...");
            
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            try {
                cap = open_cap(pipeline);
            } catch (...) {
                spdlog::error("reconexión fallida");
                break;
            }
            continue;
        }
        
        writer.write(frame);
        frames_recorded++;
        
        // Preview opcional
        if (show_preview) {
            cv::Mat display;
            cv::resize(frame, display, cv::Size(640, 480));
            
            // Overlay con info
            cv::putText(display, "REC", cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::putText(display, "Frames: " + std::to_string(frames_recorded), 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 255, 255), 1);
            
            cv::imshow("Recording", display);
            cv::waitKey(1);
        }
        
        // Verificar duración
        if (duration_seconds > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (seconds >= duration_seconds) {
                spdlog::info("duración alcanzada. deteniendo...");
                break;
            }
        }
        
        // Log cada 5 segundos
        auto now = std::chrono::steady_clock::now();
        auto log_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count();
        if (log_elapsed >= 5) {
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            spdlog::info("tiempo: {}s | frames: {} | perdidos: {}", 
                        total_elapsed, frames_recorded, frames_lost);
            last_log = now;
        }
    }
    
    // Estadísticas finales
    auto end_time = std::chrono::steady_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - start_time).count();
    
    spdlog::info("=== grabación finalizada ===");
    spdlog::info("archivo: {}", filename);
    spdlog::info("duración: {:.2f} s", total_duration);
    spdlog::info("frames grabados: {}", frames_recorded);
    spdlog::info("frames perdidos: {}", frames_lost);
    spdlog::info("fps promedio: {:.2f}", frames_recorded / total_duration);
    
    writer.release();
    cap.release();
    if (show_preview) {
        cv::destroyAllWindows();
    }
    
    return 0;
}