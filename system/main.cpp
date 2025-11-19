// ============= main.cpp =============
#include "detector_optimized.hpp"
#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <map>
#include <csignal>
#include <atomic>

// ==================== TOML PARSER ====================
class SimpleToml {
private:
    std::map<std::string, std::string> values;
    
    std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        auto end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }
    
public:
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            spdlog::warn("No se pudo abrir: {}", filename);
            return false;
        }
        
        std::string line, section;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            
            if (line[0] == '[' && line.back() == ']') {
                section = line.substr(1, line.length() - 2);
                continue;
            }
            
            auto eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = trim(line.substr(0, eq));
                std::string val = trim(line.substr(eq + 1));
                
                if (val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.length() - 2);
                }
                
                std::string full_key = section.empty() ? key : section + "." + key;
                values[full_key] = val;
            }
        }
        return true;
    }
    
    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = values.find(key);
        return it != values.end() ? it->second : def;
    }
    
    int get_int(const std::string& key, int def = 0) const {
        try { return std::stoi(get(key)); }
        catch (...) { return def; }
    }
    
    float get_float(const std::string& key, float def = 0.0f) const {
        try { return std::stof(get(key)); }
        catch (...) { return def; }
    }
    
    bool get_bool(const std::string& key, bool def = false) const {
        std::string val = get(key);
        return val == "true" || val == "1";
    }
};

std::atomic<bool> stop_signal(false);

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        spdlog::info("🛑 Deteniendo...");
        stop_signal = true;
    }
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::string config_file = argc >= 2 ? argv[1] : "config.toml";
    
    // ========== CARGAR CONFIG ==========
    SimpleToml config;
    if (!config.load(config_file)) {
        spdlog::error("❌ No se pudo cargar: {}", config_file);
        return 1;
    }
    
    std::string source = config.get("input.source", "main");
    std::string model_path = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    float conf_thr = config.get_float("detector.conf_threshold", 0.5f);
    float nms_thr = config.get_float("detector.nms_threshold", 0.4f);
    bool gpu_preproc = config.get_bool("detector.use_gpu_preprocessing", true);
    
    bool display = config.get_bool("output.display_enabled", true);
    int disp_w = config.get_int("output.display_width", 1280);
    int disp_h = config.get_int("output.display_height", 720);
    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_dets = config.get_bool("output.draw_detections", true);
    bool draw_lands = config.get_bool("output.draw_landmarks", true);
    
    int max_fps = config.get_int("performance.max_fps", 30);
    int skip_frames = config.get_int("performance.skip_frames", 0);
    
    // ========== DETECTAR TIPO DE INPUT ==========
    bool is_rtsp = (source == "main" || source == "sub");
    
    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║       PANTO - Face Detection           ║");
    spdlog::info("╚════════════════════════════════════════╝");
    spdlog::info("Config: {}", config_file);
    spdlog::info("Input: {}", source);
    spdlog::info("Modelo: {}", model_path);
    spdlog::info("GPU Preprocessing: {}", gpu_preproc ? "ON" : "OFF");
    
    // ========== CARGAR DETECTOR ==========
    try {
        FaceDetectorOptimized detector(model_path, gpu_preproc);
        detector.set_conf_threshold(conf_thr);
        detector.set_nms_threshold(nms_thr);
        
        spdlog::info("✓ Detector cargado");
        
        // ========== ABRIR INPUT ==========
        cv::VideoCapture cap;
        
        if (is_rtsp) {
            // Usar stream_capture para RTSP
            std::string user = config.get("camera.user", "admin");
            std::string pass = config.get("camera.pass", "Panto2025");
            std::string ip = config.get("camera.ip", "192.168.0.101");
            int port = config.get_int("camera.port", 554);
            
            StreamCapture stream(user, pass, ip, port, source);
            stream.set_stop_signal(&stop_signal);
            
            if (!stream.open()) {
                spdlog::error("❌ Error abriendo RTSP");
                return 1;
            }
            
            spdlog::info("✓ Stream conectado");
            
            // ========== LOOP RTSP ==========
            cv::Mat frame, display;
            int frame_count = 0;
            double total_ms = 0, total_det_ms = 0;
            int total_dets = 0;
            
            std::string window_name = "PANTO - " + source;
            if (display) {
                cv::namedWindow(window_name, cv::WINDOW_NORMAL);
                cv::resizeWindow(window_name, disp_w, disp_h);
            }
            
            auto start_time = std::chrono::steady_clock::now();
            
            while (stream.read(frame) && !stop_signal) {
                frame_count++;
                
                // Skip frames si está configurado
                if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0) {
                    continue;
                }
                
                // ========== DETECCIÓN ==========
                auto t1 = std::chrono::high_resolution_clock::now();
                auto detections = detector.detect(frame);
                auto t2 = std::chrono::high_resolution_clock::now();
                
                double det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_det_ms += det_ms;
                total_dets += detections.size();
                
                // ========== DISPLAY ==========
                if (display) {
                    cv::resize(frame, display, cv::Size(disp_w, disp_h));
                    
                    // Escalar coordenadas
                    float scale_x = (float)disp_w / frame.cols;
                    float scale_y = (float)disp_h / frame.rows;
                    
                    // Dibujar detecciones
                    if (draw_dets) {
                        for (const auto& det : detections) {
                            cv::Rect box_scaled(
                                det.box.x * scale_x,
                                det.box.y * scale_y,
                                det.box.width * scale_x,
                                det.box.height * scale_y
                            );
                            
                            cv::rectangle(display, box_scaled, cv::Scalar(0, 255, 0), 2);
                            
                            std::string label = cv::format("%.2f", det.confidence);
                            cv::putText(display, label, 
                                       cv::Point(box_scaled.x, box_scaled.y - 5),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                                       cv::Scalar(0, 255, 0), 2);
                            
                            if (draw_lands) {
                                for (int i = 0; i < 5; i++) {
                                    cv::Point2f pt(
                                        det.landmarks[i].x * scale_x,
                                        det.landmarks[i].y * scale_y
                                    );
                                    cv::circle(display, pt, 3, cv::Scalar(0, 0, 255), -1);
                                }
                            }
                        }
                    }
                    
                    // Info overlay
                    cv::Mat overlay = display.clone();
                    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(400, 100), 
                                 cv::Scalar(0, 0, 0), -1);
                    cv::addWeighted(overlay, 0.7, display, 0.3, 0, display);
                    
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    double sec = std::chrono::duration<double>(elapsed).count();
                    double proc_fps = frame_count / sec;
                    
                    if (draw_fps) {
                        cv::putText(display, cv::format("FPS: %.1f", proc_fps),
                                   cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 
                                   0.7, cv::Scalar(0, 255, 0), 2);
                    }
                    
                    cv::putText(display, cv::format("Detector: %.1fms", det_ms),
                               cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 
                               0.6, cv::Scalar(0, 255, 0), 2);
                    
                    cv::putText(display, cv::format("Rostros: %d", (int)detections.size()),
                               cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 
                               0.6, cv::Scalar(0, 255, 0), 2);
                    
                    cv::imshow(window_name, display);
                    
                    if (cv::waitKey(1) == 27) break;  // ESC
                }
                
                // FPS limiter
                if (max_fps > 0) {
                    int delay_ms = 1000 / max_fps;
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }
                
                // Stats cada 100 frames
                if (frame_count % 100 == 0) {
                    double avg_det_ms = total_det_ms / frame_count;
                    spdlog::info("Frame {} | Det: {:.1f}ms | Rostros: {:.1f}/frame", 
                                frame_count, avg_det_ms, 
                                (double)total_dets / frame_count);
                }
            }
            
            if (display) cv::destroyWindow(window_name);
            
            // Stats finales
            double avg_det_ms = total_det_ms / frame_count;
            spdlog::info("╔════════════════════════════════════════╗");
            spdlog::info("║            ESTADÍSTICAS                ║");
            spdlog::info("╚════════════════════════════════════════╝");
            spdlog::info("Frames procesados: {}", frame_count);
            spdlog::info("Detección promedio: {:.1f}ms ({:.1f} FPS)", 
                        avg_det_ms, 1000.0 / avg_det_ms);
            spdlog::info("Rostros totales: {} ({:.2f}/frame)", 
                        total_dets, (double)total_dets / frame_count);
            
        } else {
            // ========== VIDEO FILE ==========
            cap.open(source);
            if (!cap.isOpened()) {
                spdlog::error("❌ No se pudo abrir: {}", source);
                return 1;
            }
            
            int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
            double video_fps = cap.get(cv::CAP_PROP_FPS);
            int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            
            spdlog::info("✓ Video: {}x{} @ {:.1f}fps ({} frames)", 
                        width, height, video_fps, total_frames);
            
            // ========== LOOP VIDEO ==========
            cv::Mat frame, display;
            int frame_count = 0;
            double total_det_ms = 0;
            int total_dets = 0;
            bool paused = false;
            
            std::string window_name = "PANTO - Video";
            if (display) {
                cv::namedWindow(window_name, cv::WINDOW_NORMAL);
                cv::resizeWindow(window_name, disp_w, disp_h);
            }
            
            auto start_time = std::chrono::steady_clock::now();
            
            while (cap.read(frame) && !stop_signal) {
                frame_count++;
                
                if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0) {
                    continue;
                }
                
                if (!paused) {
                    // Detección
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto detections = detector.detect(frame);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    
                    double det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                    total_det_ms += det_ms;
                    total_dets += detections.size();
                    
                    if (display) {
                        cv::resize(frame, display, cv::Size(disp_w, disp_h));
                        
                        float scale_x = (float)disp_w / frame.cols;
                        float scale_y = (float)disp_h / frame.rows;
                        
                        if (draw_dets) {
                            for (const auto& det : detections) {
                                cv::Rect box_scaled(
                                    det.box.x * scale_x,
                                    det.box.y * scale_y,
                                    det.box.width * scale_x,
                                    det.box.height * scale_y
                                );
                                
                                cv::rectangle(display, box_scaled, cv::Scalar(0, 255, 0), 2);
                                
                                if (draw_lands) {
                                    for (int i = 0; i < 5; i++) {
                                        cv::Point2f pt(
                                            det.landmarks[i].x * scale_x,
                                            det.landmarks[i].y * scale_y
                                        );
                                        cv::circle(display, pt, 3, cv::Scalar(0, 0, 255), -1);
                                    }
                                }
                            }
                        }
                        
                        // Info
                        cv::Mat overlay = display.clone();
                        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(450, 120), 
                                     cv::Scalar(0, 0, 0), -1);
                        cv::addWeighted(overlay, 0.7, display, 0.3, 0, display);
                        
                        double progress = 100.0 * frame_count / total_frames;
                        double avg_det_ms = total_det_ms / frame_count;
                        
                        cv::putText(display, cv::format("Frame: %d/%d (%.1f%%)", 
                                   frame_count, total_frames, progress),
                                   cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, cv::Scalar(0, 255, 0), 2);
                        
                        cv::putText(display, cv::format("Detector: %.1fms (%.1f FPS)", 
                                   det_ms, 1000.0/det_ms),
                                   cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, cv::Scalar(0, 255, 0), 2);
                        
                        cv::putText(display, cv::format("Rostros: %d", (int)detections.size()),
                                   cv::Point(10, 75), cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, cv::Scalar(0, 255, 0), 2);
                        
                        cv::putText(display, cv::format("Avg: %.1fms", avg_det_ms),
                                   cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 
                                   0.5, cv::Scalar(0, 255, 0), 1);
                        
                        cv::imshow(window_name, display);
                    }
                }
                
                if (display) {
                    int key = cv::waitKey(paused ? 0 : 1);
                    if (key == 27) break;  // ESC
                    if (key == 32) paused = !paused;  // SPACE
                }
                
                if (frame_count % 100 == 0) {
                    double avg_ms = total_det_ms / frame_count;
                    spdlog::info("Frame {}/{} ({:.1f}%) | Det: {:.1f}ms | Rostros: {:.1f}/frame",
                                frame_count, total_frames, 
                                100.0 * frame_count / total_frames,
                                avg_ms, (double)total_dets / frame_count);
                }
            }
            
            if (display) cv::destroyWindow(window_name);
            
            // Stats finales
            double avg_det_ms = total_det_ms / frame_count;
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            double total_sec = std::chrono::duration<double>(elapsed).count();
            
            spdlog::info("╔════════════════════════════════════════╗");
            spdlog::info("║            ESTADÍSTICAS                ║");
            spdlog::info("╚════════════════════════════════════════╝");
            spdlog::info("Frames procesados: {}/{}", frame_count, total_frames);
            spdlog::info("Tiempo total: {:.1f}s", total_sec);
            spdlog::info("Throughput: {:.1f} FPS", frame_count / total_sec);
            spdlog::info("Detección promedio: {:.1f}ms ({:.1f} FPS)", 
                        avg_det_ms, 1000.0 / avg_det_ms);
            spdlog::info("Rostros totales: {} ({:.2f}/frame)", 
                        total_dets, (double)total_dets / frame_count);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("❌ Error: {}", e.what());
        return 1;
    }
    
    spdlog::info("✓ PANTO finalizado");
    return 0;
}