#include "detector_optimized.hpp"
#include "tracker.hpp"
#include "recognizer.hpp"
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
    bool mode_recognize = config.get_bool("mode.recognize", true);

    // Detector
    std::string model_path = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    float conf_thr = config.get_float("detector.conf_threshold", 0.5f);
    float nms_thr = config.get_float("detector.nms_threshold", 0.4f);
    bool gpu_preproc = config.get_bool("detector.use_gpu_preprocessing", true);

    // Recognizer
    std::string recognizer_path = config.get("recognizer.model_path", "models/arcface_r100.engine");
    int embedding_size = config.get_int("recognizer.embedding_size", 512);
    bool recog_gpu_preproc = config.get_bool("recognizer.use_gpu_preprocessing", true);
    float recog_threshold = config.get_float("recognizer.threshold", 0.6f);

    // Database
    std::string db_path = config.get("database.path", "database/faces.db");

    // Output
    int disp_w = config.get_int("output.display_width", 1280);
    int disp_h = config.get_int("output.display_height", 720);
    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_dets = config.get_bool("output.draw_detections", true);
    bool draw_lands = config.get_bool("output.draw_landmarks", true);
    bool draw_names = config.get_bool("output.draw_names", true);
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
    spdlog::info("Modes: detect={} record={} display={} recognize={}",
                mode_detect, mode_record, mode_display, mode_recognize);

    // ========== CARGAR DETECTOR, TRACKER Y RECOGNIZER ==========
    std::unique_ptr<FaceDetectorOptimized> detector;
    std::unique_ptr<FaceTracker> tracker;
    std::unique_ptr<FaceRecognizer> recognizer;
    std::unique_ptr<FaceDatabase> database;

    if (mode_detect) {
        try {
            spdlog::info("Cargando detector: {}", model_path);
            detector = std::make_unique<FaceDetectorOptimized>(model_path, gpu_preproc);
            detector->set_conf_threshold(conf_thr);
            detector->set_nms_threshold(nms_thr);
            spdlog::info("✓ Detector cargado (GPU preproc: {})", gpu_preproc);

            // Inicializar tracker
            tracker = std::make_unique<FaceTracker>(0.25f, 60, 2);
            spdlog::info("✓ Tracker inicializado");

            // Inicializar reconocedor si está habilitado
            if (mode_recognize) {
                spdlog::info("Cargando recognizer: {}", recognizer_path);
                recognizer = std::make_unique<FaceRecognizer>(recognizer_path, recog_gpu_preproc);
                spdlog::info("✓ Recognizer cargado (embedding: {}, GPU preproc: {})", 
                            embedding_size, recog_gpu_preproc);

                // Inicializar base de datos
                database = std::make_unique<FaceDatabase>(db_path, embedding_size, recog_threshold);
                spdlog::info("✓ Database inicializada ({} personas registradas)", 
                            database->count_persons());
            }

        } catch (const std::exception& e) {
            spdlog::error("❌ Error cargando componentes: {}", e.what());
            return 1;
        }
    }

    // ========== ABRIR INPUT ==========
    cv::VideoCapture cap;
    cv::VideoWriter writer;

    if (is_rtsp) {
        std::string pipeline = gst_pipeline(user, pass, ip, port, source);
        cap = open_cap(pipeline);
    } else {
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
    double total_track_ms = 0;
    double total_recog_ms = 0;
    int total_tracked = 0;
    int total_recognized = 0;

    auto start_time = std::chrono::steady_clock::now();

    spdlog::info("╔════════════════════════════════════════╗");
    spdlog::info("║          Procesando...                 ║");
    spdlog::info("╚════════════════════════════════════════╝");

    while (cap.read(frame) && !stop_signal) {
        frame_count++;

        if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0) {
            continue;
        }

        // ========== DETECTION + TRACKING ==========
        std::vector<TrackedFace> tracked_faces;
        double det_ms = 0, track_ms = 0, recog_ms = 0;

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
                    // Solo reconocer si es un track confirmado y no está ya reconocido
                    if (face.hits >= 3 && !face.is_recognized) {
                        // Extraer ROI del rostro
                        cv::Rect safe_box = face.box & cv::Rect(0, 0, frame.cols, frame.rows);
                        if (safe_box.area() > 0) {
                            cv::Mat face_roi = frame(safe_box);
                            
                            // Extraer embedding
                            auto embedding = recognizer->extract_embedding(face_roi);
                            
                            // Buscar en base de datos
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

            if (mode_detect && draw_dets) {
                for (const auto& face : tracked_faces) {
                    cv::Rect box_scaled(
                        face.box.x * scale_x,
                        face.box.y * scale_y,
                        face.box.width * scale_x,
                        face.box.height * scale_y
                    );

                    // Color según estado
                    cv::Scalar color;
                    if (face.is_recognized) {
                        color = cv::Scalar(0, 255, 0);  // Verde = reconocido
                    } else if (face.hits < 3) {
                        color = cv::Scalar(0, 255, 255);  // Amarillo = tracking inicial
                    } else {
                        color = cv::Scalar(255, 0, 0);  // Azul = tracking confirmado
                    }

                    cv::rectangle(display, box_scaled, color, 2);

                    // Label con ID y nombre
                    std::string label = cv::format("ID:%d", face.id);
                    if (face.is_recognized && draw_names) {
                        label += " - " + face.name;
                    }
                    label += cv::format(" (%.2f)", face.confidence);

                    cv::putText(display, label,
                               cv::Point(box_scaled.x, box_scaled.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

                    // Info adicional
                    std::string info = cv::format("Age:%d Hits:%d", face.age, face.hits);
                    cv::putText(display, info,
                               cv::Point(box_scaled.x, box_scaled.y + box_scaled.height + 15),
                               cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

                    // Landmarks
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

            // Panel de información
            cv::Mat overlay = display.clone();
            int panel_h = mode_recognize ? 180 : 150;
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(450, panel_h),
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

                cv::putText(display, cv::format("Tracker: %.1fms", track_ms),
                           cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                           0.6, cv::Scalar(0, 255, 0), 2);
                y += 25;

                if (mode_recognize) {
                    cv::putText(display, cv::format("Recognition: %.1fms", recog_ms),
                               cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                               0.6, cv::Scalar(0, 255, 0), 2);
                    y += 25;
                }

                cv::putText(display, cv::format("Tracked: %d", (int)tracked_faces.size()),
                           cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                           0.6, cv::Scalar(0, 255, 0), 2);
                y += 25;

                if (tracker) {
                    cv::putText(display, cv::format("Total IDs: %d", tracker->get_total_tracks()),
                               cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                               0.6, cv::Scalar(0, 255, 0), 2);
                }
            }

            if (mode_record) {
                cv::circle(display, cv::Point(430, 20), 8, cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "REC", cv::Point(390, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            }

            cv::imshow(window_name, display);

            if (cv::waitKey(1) == 27) break;  // ESC
        }

        if (max_fps > 0) {
            int delay_ms = 1000 / max_fps;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        if (frame_count % fps_interval == 0) {
            double avg_det_ms = mode_detect ? total_det_ms / frame_count : 0;
            double avg_track_ms = mode_detect ? total_track_ms / frame_count : 0;
            double avg_recog_ms = mode_recognize ? total_recog_ms / frame_count : 0;

            if (mode_detect) {
                spdlog::info("Frame {} | Det: {:.1f}ms | Track: {:.1f}ms | Recog: {:.1f}ms | Rostros: {:.1f}/frame",
                            frame_count, avg_det_ms, avg_track_ms, avg_recog_ms,
                            (double)total_tracked / frame_count);
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
        double avg_track_ms = total_track_ms / frame_count;
        spdlog::info("Detección promedio: {:.1f}ms ({:.1f} FPS)",
                    avg_det_ms, 1000.0 / avg_det_ms);
        spdlog::info("Tracking promedio: {:.1f}ms", avg_track_ms);
        
        if (mode_recognize) {
            double avg_recog_ms = total_recog_ms / frame_count;
            spdlog::info("Reconocimiento promedio: {:.1f}ms", avg_recog_ms);
            spdlog::info("Total pipeline: {:.1f}ms ({:.1f} FPS)",
                        avg_det_ms + avg_track_ms + avg_recog_ms,
                        1000.0 / (avg_det_ms + avg_track_ms + avg_recog_ms));
            spdlog::info("Rostros reconocidos: {} únicos", total_recognized);
        } else {
            spdlog::info("Total pipeline: {:.1f}ms ({:.1f} FPS)",
                        avg_det_ms + avg_track_ms,
                        1000.0 / (avg_det_ms + avg_track_ms));
        }
        
        spdlog::info("Rostros rastreados: {} ({:.2f}/frame)",
                    total_tracked, (double)total_tracked / frame_count);

        if (tracker) {
            spdlog::info("IDs únicos generados: {}", tracker->get_total_tracks());
        }
    }

    spdlog::info("✓ PANTO finalizado");
    return 0;
}