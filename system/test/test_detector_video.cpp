// test/test_detector_video.cpp - Versión con debug visual mejorado
#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);  // DEBUG para ver decodificación
    
    if (argc < 3) {
        spdlog::error("uso: {} <modelo.onnx> <video.mp4> [threshold]", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string video_path = argv[2];
    float threshold = argc >= 4 ? std::atof(argv[3]) : 0.5f;
    
    spdlog::info("=== test detector con video ===");
    spdlog::info("modelo: {}", model_path);
    spdlog::info("video: {}", video_path);
    spdlog::info("threshold: {:.2f}", threshold);
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        spdlog::error("no se pudo abrir: {}", video_path);
        return 1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps_original = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    spdlog::info("video: {}x{} @ {:.1f} fps, {} frames", width, height, fps_original, total_frames);
    
    try {
        FaceDetector detector(model_path, true);
        detector.set_conf_threshold(threshold);
        detector.set_nms_threshold(0.3f);
        
        spdlog::info("");
        spdlog::info("controles:");
        spdlog::info("  SPACE - pausar/reanudar");
        spdlog::info("  ESC   - salir");
        spdlog::info("  S     - guardar frame");
        spdlog::info("  D     - toggle debug visual");
        spdlog::info("  +/-   - ajustar threshold");
        spdlog::info("");
        
        cv::namedWindow("detector video test", cv::WINDOW_NORMAL);
        cv::resizeWindow("detector video test", 1280, 720);
        
        cv::Mat frame;
        int frame_count = 0;
        bool paused = false;
        bool debug_visual = true;  // Activado por defecto
        int saved_count = 0;
        
        int total_detections = 0;
        double total_time_ms = 0;
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (cap.read(frame)) {
            frame_count++;
            
            if (frame_count % 100 == 0) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                double process_fps = frame_count / elapsed_sec;
                spdlog::info("progreso: {}/{} frames ({:.1f}%), {:.1f} fps", 
                            frame_count, total_frames, 
                            100.0 * frame_count / total_frames,
                            process_fps);
            }
            
            if (!paused) {
                auto t_start = std::chrono::steady_clock::now();
                auto detections = detector.detect(frame);
                auto t_end = std::chrono::steady_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                total_time_ms += ms;
                total_detections += detections.size();
                
                cv::Mat display = frame.clone();
                
                for (size_t det_idx = 0; det_idx < detections.size(); det_idx++) {
                    const auto& det = detections[det_idx];
                    
                    // === BOX PRINCIPAL (VERDE) ===
                    cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 3);
                    
                    // === LANDMARKS CON COLORES DIFERENTES ===
                    std::vector<std::string> lm_names = {"L_eye", "R_eye", "Nose", "L_mouth", "R_mouth"};
                    cv::Scalar lm_colors[] = {
                        cv::Scalar(255, 0, 0),    // Azul - ojo izq
                        cv::Scalar(0, 255, 255),  // Amarillo - ojo der
                        cv::Scalar(255, 0, 255),  // Magenta - nariz
                        cv::Scalar(0, 0, 255),    // Rojo - boca izq
                        cv::Scalar(255, 255, 0)   // Cyan - boca der
                    };
                    
                    for (int i = 0; i < 5; i++) {
                        // Punto del landmark
                        cv::circle(display, det.landmarks[i], 6, lm_colors[i], -1);
                        cv::circle(display, det.landmarks[i], 8, cv::Scalar(255, 255, 255), 2);
                        
                        // Etiqueta del landmark (si debug activado)
                        if (debug_visual) {
                            cv::putText(display, lm_names[i],
                                       cv::Point(det.landmarks[i].x + 12, det.landmarks[i].y - 5),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.4,
                                       lm_colors[i], 1);
                        }
                    }
                    
                    // === LÍNEAS CONECTANDO LANDMARKS (DEBUG) ===
                    if (debug_visual) {
                        // Línea entre ojos
                        cv::line(display, det.landmarks[0], det.landmarks[1], 
                                cv::Scalar(200, 200, 200), 1);
                        // Línea entre boca
                        cv::line(display, det.landmarks[3], det.landmarks[4], 
                                cv::Scalar(200, 200, 200), 1);
                        // Cruz nariz
                        cv::drawMarker(display, det.landmarks[2], cv::Scalar(255, 0, 255),
                                      cv::MARKER_CROSS, 15, 2);
                    }
                    
                    // === CONFIANZA ===
                    std::string label = cv::format("%.3f (#%zu)", det.confidence, det_idx);
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                         0.7, 2, &baseline);
                    
                    // Fondo del texto
                    cv::rectangle(display,
                                 cv::Point(det.box.x, det.box.y - text_size.height - 10),
                                 cv::Point(det.box.x + text_size.width + 10, det.box.y),
                                 cv::Scalar(0, 255, 0), -1);
                    
                    cv::putText(display, label, 
                               cv::Point(det.box.x + 5, det.box.y - 5),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                               cv::Scalar(0, 0, 0), 2);
                    
                    // === INFO DETALLADA (DEBUG) ===
                    if (debug_visual) {
                        // Box info
                        std::string box_info = cv::format("Box: [%d,%d,%d,%d]",
                                                         det.box.x, det.box.y,
                                                         det.box.width, det.box.height);
                        cv::putText(display, box_info,
                                   cv::Point(det.box.x, det.box.y + det.box.height + 20),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                   cv::Scalar(0, 255, 0), 1);
                        
                        // Verificar landmarks dentro del box
                        int lm_inside = 0;
                        for (int i = 0; i < 5; i++) {
                            if (det.landmarks[i].x >= det.box.x && 
                                det.landmarks[i].x <= det.box.x + det.box.width &&
                                det.landmarks[i].y >= det.box.y && 
                                det.landmarks[i].y <= det.box.y + det.box.height) {
                                lm_inside++;
                            }
                        }
                        
                        std::string lm_info = cv::format("LM inside: %d/5", lm_inside);
                        cv::Scalar lm_color = lm_inside >= 3 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                        cv::putText(display, lm_info,
                                   cv::Point(det.box.x, det.box.y + det.box.height + 40),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                   lm_color, 1);
                    }
                }
                
                // === PANEL DE INFO ===
                cv::Mat overlay = display.clone();
                int panel_height = debug_visual ? 160 : 120;
                cv::rectangle(overlay, cv::Point(0, 0), cv::Point(450, panel_height), 
                             cv::Scalar(0, 0, 0), -1);
                cv::addWeighted(overlay, 0.75, display, 0.25, 0, display);
                
                std::vector<std::string> info_lines = {
                    cv::format("Frame: %d/%d (%.1f%%)", 
                              frame_count, total_frames, 
                              100.0 * frame_count / total_frames),
                    cv::format("Inferencia: %.1f ms (%.1f fps)", ms, 1000.0/ms),
                    cv::format("Rostros: %d (threshold: %.2f)", 
                              (int)detections.size(), threshold),
                    cv::format("Promedio: %.1f rostros/frame", 
                              frame_count > 0 ? (double)total_detections / frame_count : 0.0),
                    cv::format("Tiempo medio: %.1f ms", 
                              frame_count > 0 ? total_time_ms / frame_count : 0.0)
                };
                
                if (debug_visual) {
                    info_lines.push_back("Debug Visual: ON (D=toggle)");
                }
                
                if (paused) {
                    info_lines.push_back("PAUSADO (SPACE=continuar)");
                }
                
                int y = 25;
                for (const auto& line : info_lines) {
                    cv::putText(display, line, cv::Point(10, y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                               cv::Scalar(0, 255, 0), 2);
                    y += 22;
                }
                
                cv::imshow("detector video test", display);
                
            } else {
                cv::Mat display = frame.clone();
                cv::rectangle(display, cv::Point(0, 0), cv::Point(350, 35), 
                             cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "PAUSADO - SPACE=continuar", 
                           cv::Point(10, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);
                cv::imshow("detector video test", display);
            }
            
            // === CONTROLES ===
            int key = cv::waitKey(paused ? 0 : 1);
            
            if (key == 27) {  // ESC
                spdlog::info("detenido en frame {}/{}", frame_count, total_frames);
                break;
            } else if (key == 32) {  // SPACE
                paused = !paused;
                spdlog::info(paused ? "pausado" : "reanudado");
            } else if (key == 's' || key == 'S') {  // Guardar
                std::string filename = cv::format("saved_frame_%03d.jpg", saved_count++);
                cv::imwrite(filename, frame);
                spdlog::info("guardado: {}", filename);
            } else if (key == 'd' || key == 'D') {  // Debug visual
                debug_visual = !debug_visual;
                spdlog::info("debug visual: {}", debug_visual ? "ON" : "OFF");
            } else if (key == '+' || key == '=') {  // Subir threshold
                threshold = std::min(1.0f, threshold + 0.05f);
                detector.set_conf_threshold(threshold);
                spdlog::info("threshold: {:.2f}", threshold);
            } else if (key == '-' || key == '_') {  // Bajar threshold
                threshold = std::max(0.1f, threshold - 0.05f);
                detector.set_conf_threshold(threshold);
                spdlog::info("threshold: {:.2f}", threshold);
            }
        }
        
        // === STATS FINALES ===
        spdlog::info("");
        spdlog::info("=== ESTADISTICAS FINALES ===");
        spdlog::info("frames procesados: {}/{}", frame_count, total_frames);
        spdlog::info("detecciones totales: {}", total_detections);
        spdlog::info("promedio: {:.2f} rostros/frame", 
                    frame_count > 0 ? (double)total_detections / frame_count : 0.0);
        spdlog::info("tiempo promedio: {:.1f} ms/frame", 
                    frame_count > 0 ? total_time_ms / frame_count : 0.0);
        
    } catch (const std::exception& e) {
        spdlog::error("error: {}", e.what());
        return 1;
    }
    
    spdlog::info("test completado");
    return 0;
}