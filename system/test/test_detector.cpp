#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    std::string model_path = argc >= 2 ? argv[1] : "models/retinaface.onnx";
    std::string image_path = argc >= 3 ? argv[2] : "";
    
    spdlog::info("=== test detector retinaface ===");
    spdlog::info("modelo: {}", model_path);
    
    try {
        FaceDetector detector(model_path, true);
        detector.set_conf_threshold(0.5f);
        detector.set_nms_threshold(0.4f);
        
        cv::VideoCapture cap;
        cv::Mat frame;
        bool is_image_mode = false;
        
        if (!image_path.empty()) {
            // Modo imagen
            frame = cv::imread(image_path);
            if (frame.empty()) {
                spdlog::error("no se pudo leer: {}", image_path);
                return 1;
            }
            is_image_mode = true;
            spdlog::info("imagen cargada: {}x{}", frame.cols, frame.rows);
        } else {
            // Modo webcam
            cap.open(0);
            if (!cap.isOpened()) {
                spdlog::error("no se pudo abrir webcam (device_id=0)");
                spdlog::info("tip: para imagenes usa: ./test_detector model.onnx imagen.jpg");
                return 1;
            }
            spdlog::info("webcam abierta correctamente");
        }
        
        spdlog::info("presiona ESC para salir");
        
        cv::namedWindow("detector test", cv::WINDOW_NORMAL);
        
        while (true) {
            if (cap.isOpened()) {
                if (!cap.read(frame)) break;
            }
            
            // Detectar
            auto start = std::chrono::steady_clock::now();
            auto detections = detector.detect(frame);
            auto end = std::chrono::steady_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            spdlog::info("detectados: {} rostros en {:.2f}ms", detections.size(), ms);
            
            // Dibujar en frame original
            cv::Mat display = frame.clone();
            
            for (const auto& det : detections) {
                // Box
                cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 2);
                
                // Confianza
                std::string label = cv::format("%.2f", det.confidence);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                     0.5, 1, &baseline);
                
                // Fondo para texto
                cv::rectangle(display,
                             cv::Point(det.box.x, det.box.y - text_size.height - 5),
                             cv::Point(det.box.x + text_size.width, det.box.y),
                             cv::Scalar(0, 255, 0), -1);
                
                cv::putText(display, label, 
                           cv::Point(det.box.x, det.box.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 0, 0), 1);
                
                // Landmarks (ojos, nariz, boca)
                for (int i = 0; i < 5; i++) {
                    cv::circle(display, det.landmarks[i], 3, 
                              cv::Scalar(0, 0, 255), -1);
                }
            }
            
            // Info overlay
            std::string fps_text = cv::format("%.1f ms | %d faces", ms, (int)detections.size());
            cv::rectangle(display, cv::Point(0, 0), cv::Point(250, 30), 
                         cv::Scalar(0, 0, 0), -1);
            cv::putText(display, fps_text, cv::Point(10, 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                       cv::Scalar(0, 255, 0), 2);
            
            // Escalar para display si es muy grande
            cv::Mat display_scaled;
            int max_display_width = 1280;
            int max_display_height = 720;
            
            if (display.cols > max_display_width || display.rows > max_display_height) {
                double scale = std::min(
                    (double)max_display_width / display.cols,
                    (double)max_display_height / display.rows
                );
                
                int new_width = static_cast<int>(display.cols * scale);
                int new_height = static_cast<int>(display.rows * scale);
                
                cv::resize(display, display_scaled, cv::Size(new_width, new_height));
                spdlog::info("imagen escalada: {}x{} -> {}x{}", 
                            display.cols, display.rows, new_width, new_height);
            } else {
                display_scaled = display;
            }
            
            cv::imshow("detector test", display_scaled);
            
            int key = cv::waitKey(is_image_mode ? 0 : 1);
            if (key == 27) break; // ESC
            
            if (is_image_mode) break; // Imagen única, esperar ESC
        }
        
    } catch (const std::exception& e) {
        spdlog::error("error: {}", e.what());
        return 1;
    }
    
    spdlog::info("test completado");
    return 0;
}