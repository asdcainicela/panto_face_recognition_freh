#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>

// ==================== FFMPEG PIPELINE (MÁS ESTABLE) ====================
std::string ffmpeg_rtsp_url(const std::string& user, const std::string& pass, 
                            const std::string& ip, int port, const std::string& stream_type) {
    return "rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type;
}

// Legacy GStreamer pipeline (mantener por compatibilidad)
std::string gst_pipeline(const std::string& user, const std::string& pass, 
                        const std::string& ip, int port, const std::string& stream_type) {
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + 
           " latency=300 protocols=tcp buffer-mode=4 ntp-sync=true"
           " timeout=60000000 tcp-timeout=60000000 do-rtcp=true retry=20"
           " drop-on-latency=false is-live=true"
           " ! queue max-size-buffers=30 max-size-time=0 max-size-bytes=0 leaky=downstream ! "
           "rtph264depay ! h264parse config-interval=-1 ! "
           "nvv4l2decoder enable-max-performance=1 drop-frame-interval=0 ! "
           "queue max-size-buffers=10 leaky=downstream ! nvvidconv ! "
           "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=false max-buffers=20 sync=false";
}

std::string gst_pipeline_adaptive(const std::string& user, const std::string& pass, 
                                  const std::string& ip, int port, 
                                  const std::string& stream_type, double current_fps) {
    // Latencia adaptativa según FPS
    int base_latency = (stream_type == "main") ? 100 : 80;
    int adaptive_latency = base_latency;
    
    if (current_fps > 0) {
        if (current_fps <= 5) {
            adaptive_latency = static_cast<int>(base_latency * 3.0);
        }
        else if (current_fps <= 10) {
            adaptive_latency = static_cast<int>(base_latency * 2.0);
        }
        else if (current_fps <= 15) {
            adaptive_latency = static_cast<int>(base_latency * 1.5);
        }
        else if (current_fps <= 20) {
            adaptive_latency = static_cast<int>(base_latency * 1.2);
        }
        else if (current_fps <= 25) {
            adaptive_latency = base_latency;
        }
        else {
            adaptive_latency = static_cast<int>(base_latency * 0.7);
        }
        
        spdlog::debug("latencia adaptativa {} (fps={:.1f}): {}ms", 
                     stream_type, current_fps, adaptive_latency);
    }
    
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + 
           std::to_string(port) + "/" + stream_type + 
           " latency=" + std::to_string(adaptive_latency) + 
           " protocols=tcp timeout=20000000 do-rtcp=true ntp-sync=true "
           " drop-on-latency=false buffer-mode=4 tcp-timeout=20000000 retry=10 ! "
           "queue max-size-buffers=20 max-size-time=0 max-size-bytes=0 leaky=0 ! "
           "rtph264depay ! "
           "queue max-size-buffers=10 ! "
           "h264parse config-interval=-1 ! "
           "nvv4l2decoder enable-max-performance=1 enable-frame-type-reporting=1 drop-frame-interval=0 ! "
           "queue max-size-buffers=5 leaky=0 ! "
           "nvvidconv interpolation-method=5 ! "
           "video/x-raw,format=BGRx ! "
           "videoconvert ! "
           "video/x-raw,format=BGR ! "
           "appsink drop=0 max-buffers=10 sync=0";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries) {
    cv::VideoCapture cap;
    
    spdlog::info("intentando conectar...");
    
    for (int i = 0; i < retries; ++i) {
        spdlog::info("intento {}/{}...", i+1, retries);
        
        // Usar FFMPEG con opciones robustas
        cap.open(pipeline, cv::CAP_FFMPEG, {
            cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 20000,        // 20s timeout
            cv::CAP_PROP_READ_TIMEOUT_MSEC, 20000,        // 20s read timeout
            cv::CAP_PROP_BUFFERSIZE, 1                    // Buffer mínimo (como Python)
        });
        
        if (cap.isOpened()) {
            cv::Mat test_frame;
            int read_attempts = 0;
            bool can_read = false;
            
            // Dar tiempo para estabilizar
            while (read_attempts < 5 && !can_read) {
                can_read = cap.read(test_frame);
                if (!can_read) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(800));
                    read_attempts++;
                }
            }
            
            if (can_read && !test_frame.empty()) {
                double fps = cap.get(cv::CAP_PROP_FPS);
                int width = test_frame.cols;
                int height = test_frame.rows;
                
                spdlog::info("✓ conectado exitosamente");
                spdlog::info("  resolución: {}x{}", width, height);
                spdlog::info("  fps reportado: {:.1f}", fps > 0 ? fps : 0.0);
                
                return cap;
            } else {
                spdlog::warn("pipeline abierto pero no puede leer frames (intentos: {})", read_attempts);
                cap.release();
            }
        }
        
        if (i < retries - 1) {
            int wait_time = std::min(2 + i, 8);
            spdlog::warn("intento {}/{} fallido. reintentando en {}s...", i+1, retries, wait_time);
            std::this_thread::sleep_for(std::chrono::seconds(wait_time));
        }
    }
    
    spdlog::error("✗ no se pudo conectar después de {} intentos", retries);
    throw std::runtime_error("no se pudo conectar al stream RTSP");
}

bool verify_stream_health(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        return false;
    }
    
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