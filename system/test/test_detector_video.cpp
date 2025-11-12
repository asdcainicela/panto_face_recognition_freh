// test/test_detector_video.cpp - Probar detector con videos MP4
#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    if (argc < 3) {
        spdlog::error("uso: {} <modelo.onnx> <video.mp4>", argv[0]);
        spdlog::info("ejemplo: {} models/retinaface.onnx videos/recording_main_*.mp4", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string video_path = argv[2];
    
    spdlog::info("=== test detector con video ===");
    spdlog::info("modelo: {}", model_path);
    spdlog::info("video: {}", video_path);
    
    // Abrir video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        spdlog::error("no se pudo abrir: {}", video_path);
        return 1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps_original = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    spdlog::info("video: {}x{} @ {:.1f} fps, {} frames totales", 
                 width, height, fps_original, total_frames);
    
    try {
        // Cargar detector
        FaceDetector detector(model_path, true);
        detector.set_conf_threshold(0.5f);  // Threshold normal
        detector.set_nms_threshold(0.4f);
        
        spdlog::info("presiona:");
        spdlog::info("  SPACE - pausar/reanudar");
        spdlog::info("  ESC   - salir");
        spdlog::info("  S     - guardar frame actual");
        
        cv::namedWindow("detector video test", cv::WINDOW_NORMAL);
        cv::resizeWindow("detector video test", 1280, 720);
        
        cv::Mat frame;
        int frame_count = 0;
        bool paused = false;
        int saved_count = 0;
        
        // Stats
        int total_detections = 0;
        double total_time_ms = 0;
        std::vector<int> detections_per_frame;
        
        while (cap.read(frame)) {
            frame_count++;
            
            if (!paused) {
                // Detectar
                auto start = std::chrono::steady_clock::now();
                auto detections = detector.detect(frame);
                auto end = std::chrono::steady_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                total_time_ms += ms;
                total_detections += detections.size();
                detections_per_frame.push_back(detections.size());
                
                // Dibujar
                cv::Mat display = frame.clone();
                
                for (const auto& det : detections) {
                    // Box verde
                    cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 2);
                    
                    // Confianza
                    std::string label = cv::format("%.2f", det.confidence);
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                         0.6, 2, &baseline);
                    
                    cv::rectangle(display,
                                 cv::Point(det.box.x, det.box.y - text_size.height - 10),
                                 cv::Point(det.box.x + text_size.width + 5, det.box.y),
                                 cv::Scalar(0, 255, 0), -1);
                    
                    cv::putText(display, label, 
                               cv::Point(det.box.x + 2, det.box.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                               cv::Scalar(0, 0, 0), 2);
                    
                    // Landmarks (ojos, nariz, boca)
                    cv::Scalar colors[] = {
                        cv::Scalar(255, 0, 0),    // ojo izq - azul
                        cv::Scalar(0, 255, 255),  // ojo der - amarillo
                        cv::Scalar(255, 0, 255),  // nariz - magenta
                        cv::Scalar(0, 0, 255),    // boca izq - rojo
                        cv::Scalar(255, 255, 0)   // boca der - cyan
                    };
                    
                    for (int i = 0; i < 5; i++) {
                        cv::circle(display, det.landmarks[i], 4, colors[i], -1);
                    }
                }
                
                // Info panel (negro con transparencia)
                cv::Mat overlay = display.clone();
                cv::rectangle(overlay, cv::Point(0, 0), cv::Point(400, 120), 
                             cv::Scalar(0, 0, 0), -1);
                cv::addWeighted(overlay, 0.7, display, 0.3, 0, display);
                
                // Stats
                std::string info_lines[] = {
                    cv::format("Frame: %d/%d (%.1f%%)", 
                              frame_count, total_frames, 
                              100.0 * frame_count / total_frames),
                    cv::format("Inferencia: %.1f ms", ms),
                    cv::format("Detecciones: %d", (int)detections.size()),
                    cv::format("Promedio: %.1f rostros/frame", 
                              (double)total_detections / frame_count),
                    paused ? "PAUSADO (SPACE=reanudar)" : "Reproduciendo (SPACE=pausar)"
                };
                
                int y = 25;
                for (const auto& line : info_lines) {
                    cv::putText(display, line, cv::Point(10, y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                               cv::Scalar(0, 255, 0), 2);
                    y += 22;
                }
                
                cv::imshow("detector video test", display);
                
                // Log cada 30 frames
                if (frame_count % 30 == 0) {
                    spdlog::info("frame {}/{}: {} rostros, {:.1f} ms", 
                                frame_count, total_frames, detections.size(), ms);
                }
            } else {
                // Mostrar frame pausado
                cv::Mat display = frame.clone();
                cv::rectangle(display, cv::Point(0, 0), cv::Point(300, 30), 
                             cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "PAUSADO - SPACE para continuar", 
                           cv::Point(10, 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                           cv::Scalar(255, 255, 255), 2);
                cv::imshow("detector video test", display);
            }
            
            // Controles
            int key = cv::waitKey(paused ? 0 : 1);
            
            if (key == 27) {  // ESC
                spdlog::info("detenido por usuario");
                break;
            } else if (key == 32) {  // SPACE
                paused = !paused;
                spdlog::info(paused ? "pausado" : "reanudado");
            } else if (key == 's' || key == 'S') {  // Guardar frame
                std::string filename = cv::format("saved_frame_%03d.jpg", saved_count++);
                cv::imwrite(filename, frame);
                spdlog::info("guardado: {}", filename);
            }
        }
        
        // Stats finales
        spdlog::info("");
        spdlog::info("=== ESTADISTICAS FINALES ===");
        spdlog::info("frames procesados: {}/{}", frame_count, total_frames);
        spdlog::info("detecciones totales: {}", total_detections);
        spdlog::info("promedio: {:.2f} rostros/frame", (double)total_detections / frame_count);
        spdlog::info("tiempo promedio: {:.1f} ms/frame", total_time_ms / frame_count);
        spdlog::info("fps efectivo: {:.1f}", 1000.0 / (total_time_ms / frame_count));
        
        // Distribución de detecciones
        std::map<int, int> histogram;
        for (int count : detections_per_frame) {
            histogram[count]++;
        }
        
        spdlog::info("");
        spdlog::info("distribución de detecciones:");
        for (const auto& [count, freq] : histogram) {
            double percent = 100.0 * freq / frame_count;
            spdlog::info("  {} rostros: {} frames ({:.1f}%)", count, freq, percent);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("error: {}", e.what());
        return 1;
    }
    
    spdlog::info("test completado");
    return 0;
}