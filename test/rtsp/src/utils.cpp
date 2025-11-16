#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type) {
    // Latencia base más conservadora para evitar caídas
    int latency = (stream_type == "main") ? 200 : 150;
    
    // Pipeline robusto con keep-alive y manejo de errores mejorado
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + 
           " latency=" + std::to_string(latency) + 
           " protocols=tcp "           // TCP más estable que UDP
           " timeout=10000000 "        // 10 segundos timeout
           " do-rtcp=true "            // Keep-alive packets
           " ntp-sync=true "           // Sincronización temporal
           " drop-on-latency=true "    // Descartar si hay mucha latencia
           " buffer-mode=1 "           // Auto buffer management
           " retry=5 ! "               // Más reintentos
           "queue max-size-buffers=10 max-size-time=0 max-size-bytes=0 ! "
           "rtph264depay ! "
           "queue max-size-buffers=5 ! "
           "h264parse ! "
           "nvv4l2decoder enable-max-performance=1 ! "
           "queue max-size-buffers=3 leaky=2 ! "  // leaky=2 descarta frames viejos
           "nvvidconv ! video/x-raw,format=BGRx ! "
           "videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=0 max-buffers=4 sync=0 emit-signals=true";
}

std::string gst_pipeline_adaptive(const std::string& user, const std::string& pass, 
                                  const std::string& ip, int port, 
                                  const std::string& stream_type, double current_fps) {
    // Latencia dinámica basada en FPS real - Rangos extremos: 1-34 FPS
    int base_latency = (stream_type == "main") ? 200 : 150;
    int adaptive_latency = base_latency;
    
    if (current_fps > 0) {
        // FPS crítico bajo (1-5): Máxima latencia para estabilizar
        if (current_fps <= 5) {
            adaptive_latency = static_cast<int>(base_latency * 2.0);  // 400ms/300ms
        }
        // FPS muy bajo (5-10): Alta latencia
        else if (current_fps <= 10) {
            adaptive_latency = static_cast<int>(base_latency * 1.7);  // 340ms/255ms
        }
        // FPS bajo (10-15): Latencia aumentada
        else if (current_fps <= 15) {
            adaptive_latency = static_cast<int>(base_latency * 1.4);  // 280ms/210ms
        }
        // FPS medio-bajo (15-20): Latencia moderada
        else if (current_fps <= 20) {
            adaptive_latency = static_cast<int>(base_latency * 1.1);  // 220ms/165ms
        }
        // FPS medio (20-25): Latencia base
        else if (current_fps <= 25) {
            adaptive_latency = base_latency;  // 200ms/150ms
        }
        // FPS alto (25-30): Reducir latencia
        else if (current_fps <= 30) {
            adaptive_latency = static_cast<int>(base_latency * 0.8);  // 160ms/120ms
        }
        // FPS muy alto (>30): Mínima latencia
        else {
            adaptive_latency = static_cast<int>(base_latency * 0.6);  // 120ms/90ms
        }
        
        spdlog::debug("latencia adaptativa para {} (fps={:.1f}): {}ms", 
                     stream_type, current_fps, adaptive_latency);
    }
    
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + 
           " latency=" + std::to_string(adaptive_latency) + 
           " protocols=tcp timeout=10000000 do-rtcp=true ntp-sync=true "
           " drop-on-latency=true buffer-mode=1 retry=5 ! "
           "queue max-size-buffers=10 max-size-time=0 max-size-bytes=0 ! "
           "rtph264depay ! queue max-size-buffers=5 ! h264parse ! "
           "nvv4l2decoder enable-max-performance=1 ! "
           "queue max-size-buffers=3 leaky=2 ! "
           "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
           "video/x-raw,format=BGR ! appsink drop=0 max-buffers=4 sync=0";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries) {
    cv::VideoCapture cap;
    
    spdlog::info("intentando conectar...");
    spdlog::debug("pipeline: {}", pipeline);
    
    for (int i = 0; i < retries; ++i) {
        spdlog::info("intento {}/{}...", i+1, retries);
        
        cap.open(pipeline, cv::CAP_GSTREAMER);
        
        if (cap.isOpened()) {
            // Verificar que podemos leer frames
            cv::Mat test_frame;
            int read_attempts = 0;
            bool can_read = false;
            
            // Intentar leer hasta 3 veces (algunos streams tardan en iniciar)
            while (read_attempts < 3 && !can_read) {
                can_read = cap.read(test_frame);
                if (!can_read) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    read_attempts++;
                }
            }
            
            if (can_read && !test_frame.empty()) {
                // Obtener información del stream
                double fps = cap.get(cv::CAP_PROP_FPS);
                int width = test_frame.cols;
                int height = test_frame.rows;
                
                spdlog::info("✓ conectado exitosamente");
                spdlog::info("  resolución: {}x{}", width, height);
                spdlog::info("  fps reportado: {:.1f}", fps > 0 ? fps : 0.0);
                
                // Configurar buffer para mejor rendimiento
                cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
                
                return cap;
            } else {
                spdlog::warn("pipeline abierto pero no puede leer frames (intentos: {})", read_attempts);
                cap.release();
            }
        }
        
        if (i < retries - 1) {
            int wait_time = std::min(3 + i, 10); // Espera progresiva: 3, 4, 5... hasta 10s
            spdlog::warn("intento {}/{} fallido. reintentando en {}s...", i+1, retries, wait_time);
            std::this_thread::sleep_for(std::chrono::seconds(wait_time));
        }
    }
    
    spdlog::error("✗ no se pudo conectar después de {} intentos", retries);
    spdlog::error("posibles causas:");
    spdlog::error("  1. cámara no está encendida o no es accesible");
    spdlog::error("  2. credenciales incorrectas");
    spdlog::error("  3. ruta de stream incorrecta (/main, /sub)");
    spdlog::error("  4. firewall bloqueando puerto 554");
    spdlog::error("  5. problema de red/subnet");
    spdlog::error("  6. límite de conexiones simultáneas alcanzado");
    
    throw std::runtime_error("no se pudo conectar al stream RTSP");
}

bool verify_stream_health(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        return false;
    }
    
    // Verificar que el pipeline está activo
    try {
        cv::Mat test_frame;
        bool can_grab = cap.grab();
        if (can_grab) {
            cap.retrieve(test_frame);
            return !test_frame.empty();
        }
    } catch (...) {
        return false;
    }
    
    return false;
}