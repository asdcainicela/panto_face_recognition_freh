#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

std::string gst_pipeline(const std::string& user, const std::string& pass, const std::string& ip, int port) {
    return "rtspsrc location=rtsp://" + user + ":" + pass + "@" + ip + ":" + std::to_string(port) +
           "/main latency=50 ! "
           "rtph264depay ! h264parse ! nvv4l2decoder ! "
           "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink";
}

cv::VideoCapture open_cap(const std::string& pipeline, int retries=5) {
    cv::VideoCapture cap;
    for (int i = 0; i < retries; ++i) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            std::cout << "conectado exitosamente\n";
            return cap;
        }
        std::cerr << "intento " << (i+1) << "/" << retries << " fallido. reintentando en 2s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    throw std::runtime_error("no se pudo conectar a: " + pipeline);
}
