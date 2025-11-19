#include "detector_optimized.hpp"
#include "stream_capture.hpp"
#include "utils.hpp"
#include "draw_utils.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <map>
#include <csignal>
#include <atomic>
#include <filesystem>
#include <thread>

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
        if (!file.is_open()) return false;
        
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
    
    double get_double(const std::string& key, double def = 0.0) const {
        try { return std::stod(get(key)); }
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
    
    // Camera
    std::string user = config.get("camera.user", "admin");
    std::string pass = config.get("camera.pass", "Panto2025");
    std::string ip = config.get("camera.ip", "192.168.0.101");
    int port = config.get_int("camera.port", 554);
    
    // Input
    std::string source = config.get("input.source", "main");
    
    // Modes
    bool mode_detect = config.get_bool("mode.detect", true);
    bool mode_record = config.get_bool("mode.record", false);
    bool mode_display = config.get_bool("mode.display", true);
    
    // Detector
    std::string model_path = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    float conf_thr = config.get_float("detector.conf_threshold", 0.5f);
    float nms_thr = config.get_float("detector.nms_threshold", 0.4f);
    bool gpu_preproc = config.get_bool("detector.use_gpu_preprocessing", true);
    
    // Output
    int disp_w = config.get_int("output.display_width", 1280);
    int disp_h = config.get_int("output.display_height", 720);
    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_dets = config.get_bool("output.draw_detections", true);
    bool draw_lands = config.get_bool("output.draw_landmarks", true);
    std::string output_dir = config.get("output.output_dir", "videos");
    std::string video_codec = config.get("output.video_codec", "H264");
    double video_fps = config.get_double("output.video_fps", 25.0);
    
    // Performance
    int max_fps = config.get_int("performance.max_fps", 30);
    int skip_frames = config.get_int("performance.skip_frames", 0);
    int fps_interval = config.get_int("performance.fps_interval", 30);
    
    bool is_rtsp = (source == "main" || source == "sub");
    
    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║       PANTO Face Recognition           ║");
    spdlog::info("╚════════════════════════════════════════╝");
    spdlog::info("Config: {}", config_file);
    spdlog::info("Input: {}", source);
    spdlog::info("Modes: detect={} record={} display={}", 
                mode_detect, mode_record, mode_display);
    
    // ========== CARGAR DETECTOR (solo si es necesario) ==========
    std::unique_ptr<FaceDetectorOptimized> detector;
    if (mode_detect) {
        try {
            spdlog::info("Cargando detector: {}", model_path);
            detector = std::make_unique<FaceDetectorOptimized>(model_path, gpu_preproc);
            detector->set_conf_threshold(conf_thr);
            detector->set_nms_threshold(nms_thr);
            spdlog::info("✓ Detector cargado (GPU preproc: {})", gpu_preproc);
        } catch (const std::exception& e) {
            spdlog::error("❌ Error cargando detector: {}", e.what());
            return 1;
        }
    }
    
    // ========== ABRIR INPUT ==========
    cv::VideoCapture cap;
    cv::VideoWriter writer;
    
    if (is_rtsp) {
        // RTSP Stream
        std::string pipeline = gst_pipeline(user, pass, ip, port, source);
        cap = open_cap(pipeline);
    } else {
        // Video File
        cap.open(source);
        if (!cap.isOpened()) {
            spdlog::error("❌ No se pudo abrir: {}", source);
            return 1;
        }
        
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        spdlog::info("✓ Video: {}x{} @ {:.1f}fps ({} frames)", 
                    width, height, fps, total_frames);
    }
    
    // ========== SETUP RECORDING ==========
    if (mode_record) {
        std::filesystem::create_directories(output_dir);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << output_dir << "/output_" << source << "_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".mp4";
        
        std::string output_file = ss.str();
        
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        int fourcc = (video_codec == "H264") ? 
            cv::VideoWriter::fourcc('H', '2', '6', '4') : 
            cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        
        writer.open(output_file, fourcc, video_fps, cv::Size(width, height));
        
        if (writer.isOpened()) {
            spdlog::info("✓ Grabando: {}", output_file);
        } else {
            spdlog::warn("⚠️  No se pudo iniciar grabación");
            mode_record = false;
        }
    }
    
    // ========== SETUP DISPLAY ==========
    std::string window_name = "PANTO - " + source;
    if (mode_display) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, disp_w, disp_h);
    }
    
    // ========== MAIN LOOP ==========
    cv::Mat frame, display;
    int frame_count = 0;
    double total_det_ms = 0;
    int total_dets = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    
    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║          Procesando...                 ║");
    spdlog::info("╚════════════════════════════════════════╝");
    
    while (cap.read(frame) && !stop_signal) {
        frame_count++;
        
        // Skip frames
        if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0) {
            continue;
        }
        
        // ========== DETECTION ==========
        std::vector<Detection> detections;
        double det_ms = 0;
        
        if (mode_detect && detector) {
            auto t1 = std::chrono::high_resolution_clock::now();
            detections = detector->detect(frame);
            auto t2 = std::chrono::high_resolution_clock::now();
            
            det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            total_det_ms += det_ms;
            total_dets += detections.size();
        }
        
        // ========== RECORDING ==========
        if (mode_record && writer.isOpened()) {
            writer.write(frame);
        }
        
        // ========== DISPLAY ==========
        if (mode_display) {
            cv::resize(frame, display, cv::Size(disp_w, disp_h));
            
            float scale_x = (float)disp_w / frame.cols;
            float scale_y = (float)disp_h / frame.rows;
            
            // Draw detections
            if (mode_detect && draw_dets) {
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
            int panel_h = mode_detect ? 120 : 80;
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(400, panel_h), 
                         cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.7, display, 0.3, 0, display);
            
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            double sec = std::chrono::duration<double>(elapsed).count();
            double proc_fps = frame_count / sec;
            
            int y = 25;
            if (draw_fps) {
                cv::putText(display, cv::format("FPS: %.1f", proc_fps),
                           cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 
                           0.7, cv::Scalar(0, 255, 0), 2);
                y += 30;
            }
            
            if (mode_detect) {
                cv::putText(display, cv::format("Detector: %.1fms", det_ms),
                           cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, cv::Scalar(0, 255, 0), 2);
                y += 25;
                
                cv::putText(display, cv::format("Rostros: %d", (int)detections.size()),
                           cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, cv::Scalar(0, 255, 0), 2);
            }
            
            if (mode_record) {
                cv::circle(display, cv::Point(380, 20), 8, cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "REC", cv::Point(340, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            }
            
            cv::imshow(window_name, display);
            
            if (cv::waitKey(1) == 27) break;  // ESC
        }
        
        // FPS limiter
        if (max_fps > 0) {
            int delay_ms = 1000 / max_fps;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        
        // Stats
        if (frame_count % fps_interval == 0) {
            double avg_det_ms = mode_detect ? total_det_ms / frame_count : 0;
            if (mode_detect) {
                spdlog::info("Frame {} | Det: {:.1f}ms | Rostros: {:.1f}/frame", 
                            frame_count, avg_det_ms, 
                            (double)total_dets / frame_count);
            } else {
                spdlog::info("Frame {}", frame_count);
            }
        }
    }
    
    // ========== CLEANUP ==========
    if (mode_display) cv::destroyWindow(window_name);
    if (mode_record && writer.isOpened()) writer.release();
    
    // ========== STATS FINALES ==========
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double total_sec = std::chrono::duration<double>(elapsed).count();
    
    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║          ESTADÍSTICAS FINALES          ║");
    spdlog::info("╚════════════════════════════════════════╝");
    spdlog::info("Frames procesados: {}", frame_count);
    spdlog::info("Tiempo total: {:.1f}s", total_sec);
    spdlog::info("Throughput: {:.1f} FPS", frame_count / total_sec);
    
    if (mode_detect) {
        double avg_det_ms = total_det_ms / frame_count;
        spdlog::info("Detección promedio: {:.1f}ms ({:.1f} FPS)", 
                    avg_det_ms, 1000.0 / avg_det_ms);
        spdlog::info("Rostros totales: {} ({:.2f}/frame)", 
                    total_dets, (double)total_dets / frame_count);
    }
    
    spdlog::info("✓ PANTO finalizado");
    return 0;
}