// ============= main.cpp - VERSIÓN COMPLETA =============
#include "detector_optimized.hpp"
#include "tracker.hpp"
#include "recognizer.hpp"
#include "emotion_recognizer.hpp"      // ✅ AGREGADO
#include "age_gender_predictor.hpp"    // ✅ AGREGADO
#include "face_database.hpp"
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
#include <sstream>
#include <iomanip>

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
    bool mode_display = config.get_bool("mode.display", true);
    bool mode_recognize = config.get_bool("mode.recognize", true);

    // Detector
    std::string model_path = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    float conf_thr = config.get_float("detector.conf_threshold", 0.5f);
    float nms_thr = config.get_float("detector.nms_threshold", 0.4f);
    bool gpu_preproc = config.get_bool("detector.use_gpu_preprocessing", true);

    // Recognizer
    std::string recognizer_path = config.get("recognizer.model_path", "models/arcface_r100.engine");
    bool recog_gpu_preproc = config.get_bool("recognizer.use_gpu_preprocessing", true);
    float recog_threshold = config.get_float("recognizer.threshold", 0.6f);

    // ✅ EMOTION
    bool emotion_enabled = config.get_bool("emotion.enabled", false);
    std::string emotion_path = config.get("emotion.model_path", "models/emotion_ferplus.engine");
    int emotion_interval = config.get_int("emotion.analyze_interval", 5);

    // ✅ AGE/GENDER
    bool age_gender_enabled = config.get_bool("age_gender.enabled", false);
    std::string age_gender_path = config.get("age_gender.model_path", "models/age_gender.engine");
    int age_gender_interval = config.get_int("age_gender.analyze_interval", 10);

    // Database
    std::string db_path = config.get("database.path", "database/faces.db");

    // Output
    int disp_w = config.get_int("output.display_width", 1280);
    int disp_h = config.get_int("output.display_height", 720);
    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_dets = config.get_bool("output.draw_detections", true);
    bool draw_lands = config.get_bool("output.draw_landmarks", true);
    bool draw_names = config.get_bool("output.draw_names", true);

    // Performance
    int max_fps = config.get_int("performance.max_fps", 30);
    int skip_frames = config.get_int("performance.skip_frames", 0);
    int fps_interval = config.get_int("performance.fps_interval", 30);

    bool is_rtsp = (source == "main" || source == "sub");

    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║   PANTO Face Recognition COMPLETO      ║");
    spdlog::info("╚════════════════════════════════════════╝");
    spdlog::info("Config: {}", config_file);
    spdlog::info("Input: {}", source);
    spdlog::info("Modes: detect={} display={} recognize={}", 
                mode_detect, mode_display, mode_recognize);
    spdlog::info("Extras: emotion={} age_gender={}", emotion_enabled, age_gender_enabled);

    // ========== CARGAR MODELOS ==========
    std::unique_ptr<FaceDetectorOptimized> detector;
    std::unique_ptr<FaceTracker> tracker;
    std::unique_ptr<FaceRecognizer> recognizer;
    std::unique_ptr<EmotionRecognizer> emotion_recognizer;     // ✅ NUEVO
    std::unique_ptr<AgeGenderPredictor> age_gender_predictor;  // ✅ NUEVO
    std::unique_ptr<FaceDatabase> database;

    if (mode_detect) {
        try {
            spdlog::info("Cargando detector: {}", model_path);
            detector = std::make_unique<FaceDetectorOptimized>(model_path, gpu_preproc);
            detector->set_conf_threshold(conf_thr);
            detector->set_nms_threshold(nms_thr);
            spdlog::info("✓ Detector cargado");

            tracker = std::make_unique<FaceTracker>(0.25f, 60, 2);
            spdlog::info("✓ Tracker inicializado");

            if (mode_recognize) {
                spdlog::info("Cargando recognizer: {}", recognizer_path);
                recognizer = std::make_unique<FaceRecognizer>(recognizer_path, recog_gpu_preproc);
                spdlog::info("✓ Recognizer cargado");

                database = std::make_unique<FaceDatabase>(db_path, 512, recog_threshold);
                spdlog::info("✓ Database: {} personas", database->count_persons());
            }

            // ✅ CARGAR EMOTION
            if (emotion_enabled) {
                spdlog::info("Cargando emotion recognizer: {}", emotion_path);
                emotion_recognizer = std::make_unique<EmotionRecognizer>(emotion_path, true);
                spdlog::info("✓ Emotion recognizer cargado (interval: {})", emotion_interval);
            }

            // ✅ CARGAR AGE/GENDER
            if (age_gender_enabled) {
                spdlog::info("Cargando age/gender predictor: {}", age_gender_path);
                age_gender_predictor = std::make_unique<AgeGenderPredictor>(age_gender_path, true);
                spdlog::info("✓ Age/Gender predictor cargado (interval: {})", age_gender_interval);
            }

        } catch (const std::exception& e) {
            spdlog::error("❌ Error cargando componentes: {}", e.what());
            return 1;
        }
    }

    // ========== ABRIR INPUT CON THREAD EXCLUSIVO ==========
    std::unique_ptr<StreamCapture> stream_capture;
    cv::VideoCapture cap;

    if (is_rtsp) {
        // ✅ USAR NUEVO SISTEMA CON THREAD EXCLUSIVO
        spdlog::info("🎥 Iniciando captura con thread exclusivo...");
        stream_capture = std::make_unique<StreamCapture>(user, pass, ip, port, source);
        stream_capture->set_stop_signal(&stop_signal);
        stream_capture->set_fps_interval(fps_interval);
        
        if (!stream_capture->open()) {
            spdlog::error("❌ No se pudo abrir stream");
            return 1;
        }
        
        spdlog::info("✓ Stream abierto con captura en thread separado");
    } else {
        cap.open(source);
        if (!cap.isOpened()) {
            spdlog::error("❌ No se pudo abrir: {}", source);
            return 1;
        }
        spdlog::info("✓ Video abierto: {}", source);
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
    double total_track_ms = 0;
    double total_recog_ms = 0;
    double total_emotion_ms = 0;    // ✅ NUEVO
    double total_age_gender_ms = 0;  // ✅ NUEVO
    int total_tracked = 0;
    int total_recognized = 0;

    auto start_time = std::chrono::steady_clock::now();

    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║          Procesando...                 ║");
    spdlog::info("╚════════════════════════════════════════╝");

    while (!stop_signal) {
        // ✅ LEER FRAME (thread-safe si es RTSP)
        bool ok = is_rtsp ? stream_capture->read(frame) : cap.read(frame);
        
        if (!ok || frame.empty()) {
            if (!is_rtsp) break;  // Si es video, terminar
            continue;  // Si es RTSP, esperar reconexión
        }

        frame_count++;

        if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0) {
            continue;
        }

        // ========== DETECTION + TRACKING ==========
        std::vector<TrackedFace> tracked_faces;
        double det_ms = 0, track_ms = 0, recog_ms = 0;
        double emotion_ms = 0, age_gender_ms = 0;

        if (mode_detect && detector && tracker) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto detections = detector->detect(frame);
            auto t2 = std::chrono::high_resolution_clock::now();
            det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            tracked_faces = tracker->update(detections);
            auto t3 = std::chrono::high_resolution_clock::now();
            track_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

            total_det_ms += det_ms;
            total_track_ms += track_ms;
            total_tracked += tracked_faces.size();

            // ========== RECOGNITION ==========
            if (mode_recognize && recognizer && database && !tracked_faces.empty()) {
                auto t4 = std::chrono::high_resolution_clock::now();
                
                for (auto& face : tracked_faces) {
                    if (face.hits >= 3 && !face.is_recognized) {
                        cv::Rect safe_box = face.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe_box.area() > 0) {
                            cv::Mat face_roi = frame(safe_box);
                            
                            auto embedding = recognizer->extract_embedding(face_roi);
                            auto result = database->recognize(embedding);
                            
                            if (result.recognized) {
                                face.is_recognized = true;
                                face.name = result.name;
                                face.embedding = embedding;
                                total_recognized++;
                                
                                spdlog::info("✓ Reconocido: {} (ID:{}, sim:{:.3f})",
                                           result.name, face.id, result.similarity);
                            }
                        }
                    }
                }
                
                auto t5 = std::chrono::high_resolution_clock::now();
                recog_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
                total_recog_ms += recog_ms;
            }

            // ========== ✅ EMOTION ANALYSIS ==========
            if (emotion_enabled && emotion_recognizer && 
                !tracked_faces.empty() && frame_count % emotion_interval == 0) {
                auto t6 = std::chrono::high_resolution_clock::now();
                
                for (auto& face : tracked_faces) {
                    if (face.hits >= 3) {
                        cv::Rect safe_box = face.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe_box.area() > 0) {
                            cv::Mat face_roi = frame(safe_box);
                            auto emotion_result = emotion_recognizer->predict(face_roi);
                            
                            // Guardar en metadata (extender TrackedFace si es necesario)
                            spdlog::debug("ID:{} Emotion: {}", face.id, 
                                        emotion_result.to_string());
                        }
                    }
                }
                
                auto t7 = std::chrono::high_resolution_clock::now();
                emotion_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
                total_emotion_ms += emotion_ms;
            }

            // ========== ✅ AGE/GENDER PREDICTION ==========
            if (age_gender_enabled && age_gender_predictor && 
                !tracked_faces.empty() && frame_count % age_gender_interval == 0) {
                auto t8 = std::chrono::high_resolution_clock::now();
                
                for (auto& face : tracked_faces) {
                    if (face.hits >= 3) {
                        cv::Rect safe_box = face.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe_box.area() > 0) {
                            cv::Mat face_roi = frame(safe_box);
                            auto ag_result = age_gender_predictor->predict(face_roi);
                            
                            spdlog::debug("ID:{} Age/Gender: {}", face.id, 
                                        ag_result.to_string());
                        }
                    }
                }
                
                auto t9 = std::chrono::high_resolution_clock::now();
                age_gender_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
                total_age_gender_ms += age_gender_ms;
            }
        }

        // ========== DISPLAY ==========
        if (mode_display) {
            cv::resize(frame, display, cv::Size(disp_w, disp_h));

            float scale_x = (float)disp_w / frame.cols;
            float scale_y = (float)disp_h / frame.rows;

            // ======= DIBUJO DE DETECCIONES =======
            if (mode_detect && draw_dets) {
                for (const auto& face : tracked_faces) {
                    cv::Rect box_scaled(
                        face.box.x * scale_x,
                        face.box.y * scale_y,
                        face.box.width * scale_x,
                        face.box.height * scale_y
                    );

                    cv::Scalar color = face.is_recognized ? 
                        cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);

                    cv::rectangle(display, box_scaled, color, 2);

                    std::string label = cv::format("ID:%d", face.id);
                    if (face.is_recognized && draw_names) {
                        label += " - " + face.name;
                    }

                    cv::putText(display, label,
                            cv::Point(box_scaled.x, box_scaled.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

                    if (draw_lands) {
                        for (int i = 0; i < 5; i++) {
                            cv::Point2f pt(
                                face.landmarks[i].x * scale_x,
                                face.landmarks[i].y * scale_y
                            );
                            cv::circle(display, pt, 3, cv::Scalar(0, 0, 255), -1);
                        }
                    }
                }
            }

            // ======= PANEL DE INFORMACIÓN TRANSPARENTE =======
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            double sec = std::chrono::duration<double>(elapsed).count();
            double proc_fps = frame_count / sec;

            int panel_width = 280;
            int panel_height = 180;

            cv::Mat overlay = display.clone();
            cv::rectangle(overlay, cv::Rect(0, 0, panel_width, panel_height),
                        cv::Scalar(0, 0, 0), -1);

            // Transparencia
            cv::addWeighted(display, 0.7, overlay, 0.3, 0, display);

            // Borde del panel
            cv::rectangle(display, cv::Rect(0, 0, panel_width, panel_height),
                        cv::Scalar(0, 255, 0), 2);

            int y = 25;
            cv::Scalar color_green(0, 255, 0);
            cv::Scalar color_yellow(0, 255, 255);

            // FPS
            cv::putText(display, cv::format("FPS: %.1f", proc_fps),
                    cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, color_green, 2);
            y += 25;

            // Detector tiempo
            if (mode_detect) {
                cv::putText(display, cv::format("Detector: %.1fms", det_ms),
                        cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, color_green, 1);
                y += 20;

                if (emotion_enabled && emotion_ms > 0) {
                    cv::putText(display, cv::format("Emotion: %.1fms", emotion_ms),
                            cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                            0.5, color_green, 1);
                    y += 20;
                }

                if (age_gender_enabled && age_gender_ms > 0) {
                    cv::putText(display, cv::format("Age/Gender: %.1fms", age_gender_ms),
                            cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                            0.5, color_green, 1);
                    y += 20;
                }
            }

            // Estado de conexión RTSP
            if (is_rtsp && stream_capture) {
                auto stats = stream_capture->get_stats();
                cv::Scalar conn_color = (stats.reconnects == 0) ?
                    cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);

                std::string conn_status = (stats.reconnects == 0) ?
                    "CONECTADO" : cv::format("RECONEXIONES: %d", stats.reconnects);

                cv::putText(display, conn_status,
                        cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, conn_color, 1);
            }

            cv::imshow(window_name, display);

            if (cv::waitKey(1) == 27) break;
        }


        if (max_fps > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / max_fps));
        }

        if (frame_count % fps_interval == 0) {
            spdlog::info("Frame {} | Det: {:.1f}ms | Emotion: {:.1f}ms | Age/Gender: {:.1f}ms",
                        frame_count, det_ms, emotion_ms, age_gender_ms);
        }
    }

    // ========== CLEANUP ==========
    if (mode_display) cv::destroyWindow(window_name);
    if (stream_capture) stream_capture->release();

    // ========== STATS FINALES ==========
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double total_sec = std::chrono::duration<double>(elapsed).count();

    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║        ESTADÍSTICAS FINALES            ║");
    spdlog::info("╚════════════════════════════════════════╝");
    spdlog::info("Frames procesados: {}", frame_count);
    spdlog::info("Throughput: {:.1f} FPS", frame_count / total_sec);
    
    if (emotion_enabled) {
        spdlog::info("Emotion avg: {:.1f}ms", total_emotion_ms / frame_count);
    }
    
    if (age_gender_enabled) {
        spdlog::info("Age/Gender avg: {:.1f}ms", total_age_gender_ms / frame_count);
    }

    return 0;
}