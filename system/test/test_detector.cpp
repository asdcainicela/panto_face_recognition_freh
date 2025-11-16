#include "detector.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::set_level(spdlog::level::info);
    
    std::string engine_path = argc >= 2 ? argv[1] : "models/engines/model.engine";
    std::string image_path = argc >= 3 ? argv[2] : "";
    
    spdlog::info("TensorRT Engine: {}", engine_path);
    
    try {
        FaceDetector detector(engine_path);
        detector.set_conf_threshold(0.5f);
        detector.set_nms_threshold(0.4f);
        
        cv::VideoCapture cap;
        cv::Mat frame;
        bool is_image_mode = false;
        
        if (!image_path.empty()) {
            frame = cv::imread(image_path);
            if (frame.empty()) {
                spdlog::error("No se pudo leer: {}", image_path);
                return 1;
            }
            is_image_mode = true;
            spdlog::info("Imagen: {}x{}", frame.cols, frame.rows);
        } else {
            cap.open(0);
            if (!cap.isOpened()) {
                spdlog::error("No se pudo abrir webcam");
                return 1;
            }
            spdlog::info("Webcam OK");
        }
        
        spdlog::info("ESC para salir");
        cv::namedWindow("detector", cv::WINDOW_NORMAL);
        
        while (true) {
            if (cap.isOpened()) {
                if (!cap.read(frame)) break;
            }
            
            auto start = std::chrono::steady_clock::now();
            auto detections = detector.detect(frame);
            auto end = std::chrono::steady_clock::now();
            
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            cv::Mat display = frame.clone();
            
            for (const auto& det : detections) {
                cv::rectangle(display, det.box, cv::Scalar(0, 255, 0), 2);
                
                std::string label = cv::format("%.2f", det.confidence);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                     0.5, 1, &baseline);
                
                cv::rectangle(display,
                             cv::Point(det.box.x, det.box.y - text_size.height - 5),
                             cv::Point(det.box.x + text_size.width, det.box.y),
                             cv::Scalar(0, 255, 0), -1);
                
                cv::putText(display, label, 
                           cv::Point(det.box.x, det.box.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           cv::Scalar(0, 0, 0), 1);
                
                for (int i = 0; i < 5; i++) {
                    cv::circle(display, det.landmarks[i], 3, cv::Scalar(0, 0, 255), -1);
                }
            }
            
            std::string info = cv::format("TensorRT: %.1fms | %d faces", ms, (int)detections.size());
            cv::rectangle(display, cv::Point(0, 0), cv::Point(300, 30), 
                         cv::Scalar(0, 0, 0), -1);
            cv::putText(display, info, cv::Point(10, 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            cv::Mat display_scaled;
            if (display.cols > 1280 || display.rows > 720) {
                double scale = std::min((double)1280 / display.cols, (double)720 / display.rows);
                int new_w = static_cast<int>(display.cols * scale);
                int new_h = static_cast<int>(display.rows * scale);
                cv::resize(display, display_scaled, cv::Size(new_w, new_h));
            } else {
                display_scaled = display;
            }
            
            cv::imshow("detector", display_scaled);
            
            int key = cv::waitKey(is_image_mode ? 0 : 1);
            if (key == 27) break;
            if (is_image_mode) break;
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
    
    return 0;
}