// ============= main.cpp - VERSION COMPLETA =============
#include "detector_optimized.hpp"
#include "tracker.hpp"
#include "recognizer.hpp"
#include "emotion_recognizer.hpp"
#include "age_gender_predictor.hpp"
#include "face_database.hpp"
#include "stream_capture.hpp"
#include "utils.hpp"
#include "draw_utils.hpp"
#include "model_validator.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <map>
#include <csignal>
#include <atomic>
#include <filesystem>
#include <thread>
#include <sstream>
#include <iomanip>

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

                if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
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
        std::string v = get(key);
        return v == "true" || v == "1";
    }
};

std::atomic<bool> stop_signal(false);

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        spdlog::info("Deteniendo");
        stop_signal = true;
    }
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::string config_file = argc >= 2 ? argv[1] : "config.toml";

    SimpleToml config;
    if (!config.load(config_file)) {
        spdlog::error("No se pudo cargar {}", config_file);
        return 1;
    }

    std::string user = config.get("camera.user", "admin");
    std::string pass = config.get("camera.pass", "Panto2025");
    std::string ip = config.get("camera.ip", "192.168.0.101");
    int port = config.get_int("camera.port", 554);

    std::string source = config.get("input.source", "main");
    std::string capture_backend = config.get("input.capture_backend", "ffmpeg");

    bool mode_detect = config.get_bool("mode.detect", true);
    bool mode_display = config.get_bool("mode.display", true);
    bool mode_recognize = config.get_bool("mode.recognize", true);

    std::string model_path = config.get("detector.model_path", "models/scrfd_10g_bnkps.engine");
    float conf_thr = config.get_float("detector.conf_threshold", 0.5f);
    float nms_thr = config.get_float("detector.nms_threshold", 0.4f);
    bool gpu_preproc = config.get_bool("detector.use_gpu_preprocessing", true);

    std::string recognizer_path = config.get("recognizer.model_path", "models/arcface_r100.engine");
    bool recog_gpu_preproc = config.get_bool("recognizer.use_gpu_preprocessing", true);
    float recog_threshold = config.get_float("recognizer.threshold", 0.6f);

    bool emotion_enabled = config.get_bool("emotion.enabled", false);
    std::string emotion_path = config.get("emotion.model_path", "models/emotion_ferplus.engine");
    int emotion_interval = config.get_int("emotion.analyze_interval", 5);

    bool age_gender_enabled = config.get_bool("age_gender.enabled", false);
    std::string age_gender_path = config.get("age_gender.model_path", "models/age_gender.engine");
    int age_gender_interval = config.get_int("age_gender.analyze_interval", 10);

    ModelValidator validator;
    if (!validator.validate_all(
        model_path,
        recognizer_path,
        mode_recognize,
        emotion_path,
        emotion_enabled,
        age_gender_path,
        age_gender_enabled
    )) {
        spdlog::error("Model validation failed");
        return 1;
    }

    std::string db_path = config.get("database.path", "database/faces.db");

    int disp_w = config.get_int("output.display_width", 1280);
    int disp_h = config.get_int("output.display_height", 720);

    bool draw_fps = config.get_bool("output.draw_fps", true);
    bool draw_dets = config.get_bool("output.draw_detections", true);
    bool draw_lands = config.get_bool("output.draw_landmarks", true);
    bool draw_names = config.get_bool("output.draw_names", true);

    int max_fps = config.get_int("performance.max_fps", 30);
    int skip_frames = config.get_int("performance.skip_frames", 0);
    int fps_interval = config.get_int("performance.fps_interval", 30);

    bool is_rtsp = (source == "main" || source == "sub");

    std::unique_ptr<FaceDetectorOptimized> detector;
    std::unique_ptr<FaceTracker> tracker;
    std::unique_ptr<FaceRecognizer> recognizer;
    std::unique_ptr<EmotionRecognizer> emotion_recognizer;
    std::unique_ptr<AgeGenderPredictor> age_gender_predictor;
    std::unique_ptr<FaceDatabase> database;

    if (mode_detect) {
        try {
            detector = std::make_unique<FaceDetectorOptimized>(model_path, gpu_preproc);
            detector->set_conf_threshold(conf_thr);
            detector->set_nms_threshold(nms_thr);

            tracker = std::make_unique<FaceTracker>(0.25f, 60, 2);

            if (mode_recognize) {
                recognizer = std::make_unique<FaceRecognizer>(recognizer_path, recog_gpu_preproc);
                database = std::make_unique<FaceDatabase>(db_path, 512, recog_threshold);
            }

            if (emotion_enabled) {
                emotion_recognizer = std::make_unique<EmotionRecognizer>(emotion_path, true);
            }

            if (age_gender_enabled) {
                age_gender_predictor = std::make_unique<AgeGenderPredictor>(age_gender_path, true);
            }
        }
        catch (const std::exception& e) {
            spdlog::error("Error: {}", e.what());
            return 1;
        }
    }

    std::unique_ptr<StreamCapture> stream_capture;
    cv::VideoCapture cap;

    if (is_rtsp) {
        stream_capture = std::make_unique<StreamCapture>(
            user, pass, ip, port, source, capture_backend
        );
        stream_capture->set_stop_signal(&stop_signal);
        stream_capture->set_fps_interval(fps_interval);

        if (!stream_capture->open()) {
            spdlog::error("No se pudo abrir stream");
            return 1;
        }
    } else {
        cap.open(source);
        if (!cap.isOpened()) {
            spdlog::error("No se pudo abrir {}", source);
            return 1;
        }
    }

    std::string window_name = "PANTO - " + source;
    if (mode_display) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, disp_w, disp_h);
    }

    cv::Mat frame, display;
    int frame_count = 0;

    double det_ms = 0;
    double track_ms = 0;
    double recog_ms = 0;
    double emotion_ms = 0;
    double age_gender_ms = 0;

    auto start_time = std::chrono::steady_clock::now();

    while (!stop_signal) {
        bool ok = is_rtsp ? stream_capture->read(frame) : cap.read(frame);
        if (!ok || frame.empty()) {
            if (!is_rtsp) break;
            continue;
        }

        frame_count++;

        if (skip_frames > 0 && (frame_count % (skip_frames + 1)) != 0)
            continue;

        std::vector<TrackedFace> tracked_faces;







        if (mode_detect) {
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dets = detector->detect(frame);
            auto t2 = std::chrono::high_resolution_clock::now();
            det_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (frame_count % 30 == 0) {
                spdlog::info("📊 Frame {}: Detections={}, Det={}ms", 
                            frame_count, dets.size(), (int)det_ms);
            }

            tracked_faces = tracker->update(dets);
            auto t3 = std::chrono::high_resolution_clock::now();
            track_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

            if (frame_count % 30 == 0) {
                spdlog::info("🎯 Tracked faces: {}", tracked_faces.size());
            }

            if (mode_recognize) {
                auto t4 = std::chrono::high_resolution_clock::now();

                for (auto& f : tracked_faces) {
                    if (f.hits < 3 || f.is_recognized) continue;

                    cv::Rect safe = f.box & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (safe.area() <= 0) continue;

                    auto emb = recognizer->extract_embedding(frame(safe));
                    auto res = database->recognize(emb);

                    if (res.recognized) {
                        f.is_recognized = true;
                        f.name = res.name;
                    }
                }

                auto t5 = std::chrono::high_resolution_clock::now();
                recog_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
            }

            if (emotion_enabled && frame_count % emotion_interval == 0) {
                auto t6 = std::chrono::high_resolution_clock::now();
                for (auto& f : tracked_faces) {
                    if (f.hits < 3) continue;

                    cv::Rect safe = f.box & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (safe.area() <= 0) continue;

                    auto r = emotion_recognizer->predict(frame(safe));
                    f.emotion = emotion_to_string(r.emotion);
                    f.emotion_confidence = r.confidence;
                }
                auto t7 = std::chrono::high_resolution_clock::now();
                emotion_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
            }

            if (age_gender_enabled && frame_count % age_gender_interval == 0) {
                auto t8 = std::chrono::high_resolution_clock::now();
                for (auto& f : tracked_faces) {
                    if (f.hits < 3) continue;

                    cv::Rect safe = f.box & cv::Rect(0, 0, frame.cols, frame.rows);
                    if (safe.area() <= 0) continue;

                    auto r = age_gender_predictor->predict(frame(safe));
                    f.age_years = r.age;
                    f.gender = gender_to_string(r.gender);
                    f.gender_confidence = r.gender_confidence;
                }
                auto t9 = std::chrono::high_resolution_clock::now();
                age_gender_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
            }
        }

        if (mode_display) {
            cv::resize(frame, display, cv::Size(disp_w, disp_h));
            float sx = (float)disp_w / frame.cols;
            float sy = (float)disp_h / frame.rows;

            for (auto& f : tracked_faces) {
                cv::Rect box(
                    f.box.x * sx,
                    f.box.y * sy,
                    f.box.width * sx,
                    f.box.height * sy
                );

                cv::rectangle(display, box,
                    f.is_recognized ? cv::Scalar(0,255,0) : cv::Scalar(255,0,0),
                    2
                );

                int y = box.y - 10;
                std::string line = "ID:" + std::to_string(f.id);
                if (f.is_recognized && draw_names) line += " - " + f.name;

                cv::putText(display, line, cv::Point(box.x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1
                );

                y += 18;
                if (age_gender_enabled && f.age_years > 0) {
                    cv::putText(display,
                        std::to_string(f.age_years) + " years, " + f.gender,
                        cv::Point(box.x, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(255,255,0),
                        1
                    );
                    y += 18;
                }

                if (emotion_enabled && !f.emotion.empty() && f.emotion != "Unknown") {
                    cv::putText(
                        display, f.emotion,
                        cv::Point(box.x, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0,255,255),
                        1
                    );
                    y += 18;
                }

                if (draw_lands) {
                    for (int i = 0; i < 5; i++) {
                        cv::circle(display,
                            cv::Point(f.landmarks[i].x * sx, f.landmarks[i].y * sy),
                            2, cv::Scalar(0,0,255), -1
                        );
                    }
                }
            }

            auto elapsed = std::chrono::steady_clock::now() - start_time;
            double sec = std::chrono::duration<double>(elapsed).count();
            double fps = frame_count / sec;

            int pw = 300;
            int ph = 260;

            cv::Mat overlay = display.clone();
            cv::rectangle(overlay, cv::Rect(0,0,pw,ph), cv::Scalar(0,0,0), -1);
            cv::addWeighted(display, 0.7, overlay, 0.3, 0, display);

            int y = 25;
            cv::putText(display, "PANTO SYSTEM",
                cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(255,255,255), 1
            );
            y += 25;

            auto put = [&](const std::string& t){
                cv::putText(display, t, cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1
                );
                y += 22;
            };

            put("FPS: " + std::to_string((int)fps));
            put("Frames: " + std::to_string(frame_count));
            put("Det: " + std::to_string((int)det_ms) + " ms");
            put("Track: " + std::to_string((int)track_ms) + " ms");

            if (mode_recognize) put("Recognize: " + std::to_string((int)recog_ms) + " ms");
            if (emotion_enabled) put("Emotion: " + std::to_string((int)emotion_ms) + " ms");
            if (age_gender_enabled) put("Age/Gender: " + std::to_string((int)age_gender_ms) + " ms");

            cv::imshow(window_name, display);

            if (cv::waitKey(1) == 27) break;
        }
    }

    return 0;
}
