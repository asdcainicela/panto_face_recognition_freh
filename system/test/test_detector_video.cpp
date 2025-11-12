// test/test_detector_video.cpp - Probar detector con videos MP4
#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);  // DEBUG para ver todo
    
    if (argc < 3) {
        spdlog::error("uso: {} <modelo.onnx> <video.mp4> [threshold]", argv[0]);
        spdlog::info("ejemplo: {} models/retinaface.onnx videos/recording.mp4 0.5", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string video_path = argv[2];
    float threshold = argc >= 4 ? std::atof(argv[3]) : 0.5f;
    
    spdlog::info("=== test detector con video ===");
    spdlog::info("modelo: {}", model_path);
    spdlog::info("video: {}", video_path);
    spdlog::info("threshold: {:.2f}", threshold);
    
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
        detector.set_conf_threshold(threshold);
        detector.set_nms_threshold(0.3f);
        
        spdlog::info("");
        spdlog::info("controles:");
        spdlog::info("  SPACE - pausar/reanudar");
        spdlog::info("  ESC   - salir");
        spdlog::info("  S     - guardar frame");
        spdlog::info("  +/-   - ajustar threshold");
        spdlog::info("  F     - procesar rápido (sin display)");
        spdlog::info("");
        
        cv::namedWindow("detector video test", cv::WINDOW_NORMAL);
        cv::resizeWindow("detector video test", 1280, 720);
        
        cv::Mat frame;
        int frame_count = 0;
        bool paused = false;
        bool fast_mode = false;
        int saved_count = 0;
        
        // Stats
        int total_detections = 0;
        double total_time_ms = 0;
        std::vector<int> detections_per_frame;
        std::vector<double> times_per_frame;
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (cap.read(frame)) {
            frame_count++;
            
            // Debug: verificar que el frame es válido
            if (frame_count == 1) {
                spdlog::info("primer frame: {}x{}, channels={}, type={}", 
                            frame.cols, frame.rows, frame.channels(), frame.type());
                
                // Verificar que no está vacío o corrupto
                if (frame.empty()) {
                    spdlog::error("frame vacío!");
                    break;
                }
                
                if (frame.channels() != 3) {
                    spdlog::warn("frame no es BGR (channels={})", frame.channels());
                }
            }
            
            if (frame_count % 100 == 0) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                double process_fps = frame_count / elapsed_sec;
                spdlog::info("progreso: {}/{} frames ({:.1f}%), {:.1f} fps procesamiento", 
                            frame_count, total_frames, 
                            100.0 * frame_count / total_frames,
                            process_fps);
            }
            
            if (!paused) {
                // Detectar
                auto t_start = std::chrono::steady_clock::now();
                auto detections = detector.detect(frame);
                auto t_end = std::chrono::steady_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                total_time_ms += ms;
                total_detections += detections.size();
                detections_per_frame.push_back(detections.size());
                times_per_frame.push_back(ms);
                
                // Dibujar solo si no está en fast mode
                if (!fast_mode) {
                    cv::Mat display = frame.clone();
                    
                    for (const auto& det : detections) {
                        // Box verde
                        cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 3);
                        
                        // Confianza
                        std::string label = cv::format("%.2f", det.confidence);
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
                        
                        // Landmarks (ojos, nariz, boca)
                        cv::Scalar colors[] = {
                            cv::Scalar(255, 0, 0),    // ojo izq
                            cv::Scalar(0, 255, 255),  // ojo der
                            cv::Scalar(255, 0, 255),  // nariz
                            cv::Scalar(0, 0, 255),    // boca izq
                            cv::Scalar(255, 255, 0)   // boca der
                        };
                        
                        for (int i = 0; i < 5; i++) {
                            cv::circle(display, det.landmarks[i], 5, colors[i], -1);
                        }
                    }
                    
                    // Info panel
                    cv::Mat overlay = display.clone();
                    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(450, 140), 
                                 cv::Scalar(0, 0, 0), -1);
                    cv::addWeighted(overlay, 0.75, display, 0.25, 0, display);
                    
                    // Stats
                    std::vector<std::string> info_lines = {
                        cv::format("Frame: %d/%d (%.1f%%)", 
                                  frame_count, total_frames, 
                                  100.0 * frame_count / total_frames),
                        cv::format("Inferencia: %.1f ms (%.1f fps)", ms, 1000.0/ms),
                        cv::format("Rostros: %d (threshold: %.2f)", 
                                  (int)detections.size(), threshold),
                        cv::format("Promedio: %.1f rostros/frame", 
                                  (double)total_detections / frame_count),
                        cv::format("Tiempo medio: %.1f ms", total_time_ms / frame_count)
                    };
                    
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
                }
            } else {
                // Mostrar frame pausado
                cv::Mat display = frame.clone();
                cv::rectangle(display, cv::Point(0, 0), cv::Point(350, 35), 
                             cv::Scalar(0, 0, 255), -1);
                cv::putText(display, "PAUSADO - SPACE=continuar", 
                           cv::Point(10, 25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);
                cv::imshow("detector video test", display);
            }
            
            // Controles
            int key = cv::waitKey(fast_mode ? 1 : (paused ? 0 : 1));
            
            if (key == 27) {  // ESC
                spdlog::info("detenido por usuario en frame {}/{}", frame_count, total_frames);
                break;
            } else if (key == 32) {  // SPACE
                paused = !paused;
                spdlog::info(paused ? "pausado" : "reanudado");
            } else if (key == 's' || key == 'S') {  // Guardar
                std::string filename = cv::format("saved_frame_%03d.jpg", saved_count++);
                cv::imwrite(filename, frame);
                spdlog::info("guardado: {}", filename);
            } else if (key == 'f' || key == 'F') {  // Fast mode
                fast_mode = !fast_mode;
                spdlog::info(fast_mode ? "modo rápido activado" : "modo normal");
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
        
        // Stats finales
        spdlog::info("");
        spdlog::info("=== ESTADISTICAS FINALES ===");
        spdlog::info("frames procesados: {}/{}", frame_count, total_frames);
        spdlog::info("detecciones totales: {}", total_detections);
        spdlog::info("promedio: {:.2f} rostros/frame", (double)total_detections / frame_count);
        spdlog::info("tiempo promedio: {:.1f} ms/frame", total_time_ms / frame_count);
        spdlog::info("fps efectivo: {:.1f}", 1000.0 / (total_time_ms / frame_count));
        
        // Percentiles de tiempo
        std::sort(times_per_frame.begin(), times_per_frame.end());
        int p50_idx = times_per_frame.size() / 2;
        int p95_idx = static_cast<int>(times_per_frame.size() * 0.95);
        int p99_idx = static_cast<int>(times_per_frame.size() * 0.99);
        
        spdlog::info("");
        spdlog::info("latencias:");
        spdlog::info("  P50: {:.1f} ms", times_per_frame[p50_idx]);
        spdlog::info("  P95: {:.1f} ms", times_per_frame[p95_idx]);
        spdlog::info("  P99: {:.1f} ms", times_per_frame[p99_idx]);
        spdlog::info("  Max: {:.1f} ms", times_per_frame.back());
        
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
        
        // Frames con más detecciones (posibles problemas)
        if (!detections_per_frame.empty()) {
            auto max_det = *std::max_element(detections_per_frame.begin(), detections_per_frame.end());
            if (max_det > 3) {
                spdlog::warn("");
                spdlog::warn("max detecciones: {} (posibles falsos positivos)", max_det);
                spdlog::warn("considera subir threshold a {:.2f}", threshold + 0.1f);
            }
        }
        
    } catch (const std::exception& e) {
        spdlog::error("error: {}", e.what());
        return 1;
    }
    
    spdlog::info("");
    spdlog::info("test completado");
    return 0;
}