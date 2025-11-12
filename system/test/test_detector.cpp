#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);
    
    std::string model_path = argc >= 2 ? argv[1] : "models/retinaface.onnx";
    std::string image_path = argc >= 3 ? argv[2] : "";
    
    spdlog::info("=== test detector retinaface ===");
    spdlog::info("modelo: {}", model_path);
    
    try {
        FaceDetector detector(model_path, true);
        detector.set_conf_threshold(0.6f);
        
        cv::VideoCapture cap;
        cv::Mat frame;
        
        if (!image_path.empty()) {
            // Modo imagen
            frame = cv::imread(image_path);
            if (frame.empty()) {
                spdlog::error("no se pudo leer: {}", image_path);
                return 1;
            }
        } else {
            // Modo webcam
            cap.open(0);
            if (!cap.isOpened()) {
                spdlog::error("no se pudo abrir webcam");
                return 1;
            }
        }
        
        spdlog::info("presiona ESC para salir");
        
        while (true) {
            if (cap.isOpened()) {
                if (!cap.read(frame)) break;
            }
            
            auto start = std::chrono::steady_clock::now();
            auto detections = detector.detect(frame);
            auto end = std::chrono::steady_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            spdlog::info("detectados: {} rostros en {:.2f}ms", detections.size(), ms);
            
            // Dibujar
            for (const auto& det : detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                
                std::string label = cv::format("%.2f", det.confidence);
                cv::putText(frame, label, 
                           cv::Point(det.box.x, det.box.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 255, 0), 2);
                
                // Landmarks
                for (int i = 0; i < 5; i++) {
                    cv::circle(frame, det.landmarks[i], 2, 
                              cv::Scalar(0, 0, 255), -1);
                }
            }
            
            // FPS
            std::string fps_text = cv::format("%.2f ms/frame", ms);
            cv::putText(frame, fps_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(255, 0, 0), 2);
            
            cv::imshow("detector test", frame);
            
            int key = cv::waitKey(image_path.empty() ? 1 : 0);
            if (key == 27) break; // ESC
            
            if (!image_path.empty()) break; // Imagen única
        }
        
    } catch (const std::exception& e) {
        spdlog::error("error: {}", e.what());
        return 1;
    }
    
    spdlog::info("test completado");
    return 0;
}