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

int main() {
    auto start_main = std::chrono::steady_clock::now();
    std::string user = "admin", pass = "Panto2025", ip = "192.168.0.101";
    int port = 554;
    std::string pipeline = gst_pipeline(user, pass, ip, port);

    cv::VideoCapture cap;
    try {
        cap = open_cap(pipeline);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return -1;
    }

    cv::Mat frame, display;
    int frames = 0, lost = 0;
    auto start_fps = std::chrono::steady_clock::now();

    std::string window_name = "rtsp " + ip + "/main " + std::to_string(port) + " stream";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    while (true) {
        if (!cap.read(frame)) {
            lost++;
            std::cerr << "frame perdido. reconectando...\n";
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            try {
                cap = open_cap(pipeline);
            } catch (...) {
                std::cerr << "reconexion fallida\n";
                break;
            }
            continue;
        }

        frames++;
        cv::resize(frame, display, cv::Size(640, 360));
        cv::imshow(window_name, display);

        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;

        if (frames % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            double fps = 30.0 / std::chrono::duration<double>(now - start_fps).count();
            start_fps = now;
            std::cout << "frames: " << frames
                      << " | fps: " << int(fps)
                      << " | perdidos: " << lost << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();

    auto end_main = std::chrono::steady_clock::now();
    std::cout << "\n=== estadisticas finales ===\n";
    std::cout << "duracion total: " << std::chrono::duration<double>(end_main - start_main).count() << " s\n";
    std::cout << "frames totales: " << frames << " | frames perdidos: " << lost << "\n";
    std::cout << "fps promedio: " << frames / std::chrono::duration<double>(end_main - start_main).count() << "\n";
    return 0;
}
