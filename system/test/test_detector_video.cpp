#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::set_level(spdlog::level::info);
    
    if (argc < 3) {
        spdlog::error("Uso: {} <modelo.onnx> <video.mp4> [threshold]", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string video_path = argv[2];
    float threshold = argc >= 4 ? std::atof(argv[3]) : 0.5f;
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        spdlog::error("No se pudo abrir: {}", video_path);
        return 1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    spdlog::info("Video: {}x{} @ {:.1f}fps, {} frames", width, height, fps, total_frames);
    
    try {
        FaceDetector detector(model_path, true);
        detector.set_conf_threshold(threshold);
        
        spdlog::info("Controles: SPACE=pausar | ESC=salir | +/-=threshold | S=guardar");
        
        cv::namedWindow("detector", cv::WINDOW_NORMAL);
        cv::resizeWindow("detector", 1280, 720);
        
        cv::Mat frame;
        int frame_count = 0;
        bool paused = false;
        int saved = 0;
        
        int total_dets = 0;
        double total_ms = 0;
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (cap.read(frame)) {
            frame_count++;
            
            if (frame_count % 100 == 0) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double sec = std::chrono::duration<double>(elapsed).count();
                double proc_fps = frame_count / sec;
                spdlog::info("Frame {}/{} ({:.1f}%) @ {:.1f}fps", 
                            frame_count, total_frames, 
                            100.0 * frame_count / total_frames, proc_fps);
            }
            
            if (!paused) {
                auto t1 = std::chrono::steady_clock::now();
                auto dets = detector.detect(frame);
                auto t2 = std::chrono::steady_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                total_ms += ms;
                total_dets += dets.size();
                
                cv::Mat display = frame.clone();
                
                for (const auto& det : dets) {
                    cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 3);
                    
                    for (int i = 0; i < 5; i++) {
                        cv::circle(display, det.landmarks[i], 6, cv::Scalar(0, 0, 255), -1);
                    }
                    
                    std::string label = cv::format("%.3f", det.confidence);
                    cv::rectangle(display,
                                 cv::Point(det.box.x, det.box.y - 25),
                                 cv::Point(det.box.x + 100, det.box.y),
                                 cv::Scalar(0, 255, 0), -1);
                    cv::putText(display, label, 
                               cv::Point(det.box.x + 5, det.box.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                }
                
                cv::Mat overlay = display.clone();
                cv::rectangle(overlay, cv::Point(0, 0), cv::Point(450, 120), 
                             cv::Scalar(0, 0, 0), -1);
                cv::addWeighted(overlay, 0.75, display, 0.25, 0, display);
                
                std::vector<std::string> lines = {
                    cv::format("Frame: %d/%d (%.1f%%)", frame_count, total_frames, 
                              100.0 * frame_count / total_frames),
                    cv::format("Inferencia: %.1fms (%.1ffps)", ms, 1000.0/ms),
                    cv::format("Rostros: %d (thr: %.2f)", (int)dets.size(), threshold),
                    cv::format("Promedio: %.1f rostros/frame", 
                              frame_count > 0 ? (double)total_dets / frame_count : 0.0),
                    cv::format("Tiempo: %.1fms/frame", 
                              frame_count > 0 ? total_ms / frame_count : 0.0)
                };
                
                int y = 25;
                for (const auto& line : lines) {
                    cv::putText(display, line, cv::Point(10, y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    y += 22;
                }
                
                cv::imshow("detector", display);
            } else {
                cv::Mat display = frame.clone();
                cv::rectangle(display, cv::Point(0, 0), cv::Point(300, 35), 
                             cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "PAUSADO - SPACE=continuar", 
                           cv::Point(10, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                cv::imshow("detector", display);
            }
            
            int key = cv::waitKey(paused ? 0 : 1);
            
            if (key == 27) {
                spdlog::info("Detenido en frame {}/{}", frame_count, total_frames);
                break;
            } else if (key == 32) {
                paused = !paused;
            } else if (key == 's' || key == 'S') {
                std::string fn = cv::format("saved_%03d.jpg", saved++);
                cv::imwrite(fn, frame);
                spdlog::info("Guardado: {}", fn);
            } else if (key == '+' || key == '=') {
                threshold = std::min(1.0f, threshold + 0.05f);
                detector.set_conf_threshold(threshold);
                spdlog::info("Threshold: {:.2f}", threshold);
            } else if (key == '-' || key == '_') {
                threshold = std::max(0.1f, threshold - 0.05f);
                detector.set_conf_threshold(threshold);
                spdlog::info("Threshold: {:.2f}", threshold);
            }
        }
        
        spdlog::info("=== Stats ===");
        spdlog::info("Frames: {}/{}", frame_count, total_frames);
        spdlog::info("Detecciones: {} ({:.2f}/frame)", total_dets, 
                    frame_count > 0 ? (double)total_dets / frame_count : 0.0);
        spdlog::info("Tiempo: {:.1f}ms/frame", 
                    frame_count > 0 ? total_ms / frame_count : 0.0);
        
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
    
    return 0;
}