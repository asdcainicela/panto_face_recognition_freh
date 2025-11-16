#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type) {
    // Latencia más alta = menos artifacts, más smooth
    int latency = (stream_type == "main") ? 200 : 150;
    
    // Pipeline optimizado para calidad visual
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + 
           " latency=" + std::to_string(latency) + 
           " buffer-mode=1 "  // auto buffer management
           "protocols=tcp "    // TCP más estable que UDP
           "timeout=5000000 retry=3 ! "
           "queue max-size-buffers=10 max-size-time=0 max-size-bytes=0 ! "  // Buffer intermedio
           "rtph264depay ! "
           "queue max-size-buffers=5 ! "  // Otro buffer para suavizar
           "h264parse ! "
           "nvv4l2decoder enable-max-performance=1 ! "  // Máximo rendimiento del decoder
           "queue max-size-buffers=3 ! "
           "nvvidconv ! video/x-raw,format=BGRx ! "
           "videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=0 max-buffers=4 sync=0";  // No descartar frames, más buffers
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries) {
    cv::VideoCapture cap;
    
    spdlog::info("intentando conectar con pipeline:");
    spdlog::info("{}", pipeline);
    
    for (int i = 0; i < retries; ++i) {
        spdlog::info("intento {}/{}...", i+1, retries);
        
        cap.open(pipeline, cv::CAP_GSTREAMER);
        
        if (cap.isOpened()) {
            // Verify we can actually read frames
            cv::Mat test_frame;
            bool can_read = cap.read(test_frame);
            
            if (can_read && !test_frame.empty()) {
                // Obtener información del stream
                double fps = cap.get(cv::CAP_PROP_FPS);
                int width = test_frame.cols;
                int height = test_frame.rows;
                
                spdlog::info("✓ conectado exitosamente");
                spdlog::info("  resolución: {}x{}", width, height);
                spdlog::info("  fps reportado: {:.1f}", fps > 0 ? fps : 0.0);
                
                // Configurar buffer settings para mejor rendimiento
                cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
                
                return cap;
            } else {
                spdlog::warn("pipeline abierto pero no puede leer frames");
                cap.release();
            }
        }
        
        if (i < retries - 1) {
            spdlog::warn("intento {}/{} fallido. reintentando en 3s...", i+1, retries);
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    
    spdlog::error("✗ no se pudo conectar después de {} intentos", retries);
    spdlog::error("posibles causas:");
    spdlog::error("  1. cámara no está encendida o no es accesible");
    spdlog::error("  2. credenciales incorrectas");
    spdlog::error("  3. ruta de stream incorrecta (/main, /sub)");
    spdlog::error("  4. firewall bloqueando puerto 554");
    spdlog::error("  5. problema de red/subnet");
    
    throw std::runtime_error("no se pudo conectar al stream RTSP");
}

// Nueva función de utilidad para verificar conectividad
bool test_rtsp_connectivity(const std::string& user, const std::string& pass,
                            const std::string& ip, int port) {
    spdlog::info("=== test de conectividad RTSP ===");
    spdlog::info("host: {}:{}", ip, port);
    
    // Probar streams comunes
    std::vector<std::string> stream_paths = {"main", "sub", "stream1", "stream2", "h264", "live"};
    
    for (const auto& path : stream_paths) {
        std::string test_pipeline = gst_pipeline(user, pass, ip, port, path);
        cv::VideoCapture test_cap(test_pipeline, cv::CAP_GSTREAMER);
        
        if (test_cap.isOpened()) {
            cv::Mat frame;
            if (test_cap.read(frame) && !frame.empty()) {
                spdlog::info("✓ stream '{}' disponible ({}x{})", path, frame.cols, frame.rows);
                test_cap.release();
                return true;
            }
        }
        test_cap.release();
    }
    
    spdlog::error("✗ ningún stream disponible");
    return false;
}