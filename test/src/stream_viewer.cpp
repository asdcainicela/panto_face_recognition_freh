
#include "stream_viewer.hpp"
#include "utils.hpp"
#include <iostream>
#include <thread>

StreamViewer::StreamViewer(const std::string& user, const std::string& pass, 
                           const std::string& ip, int port, const std::string& stream_type,
                           cv::Size display_size, int fps_interval)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      display_size(display_size), fps_interval(fps_interval),
      frames(0), lost(0) {
    
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    window_name = "rtsp " + ip + "/" + stream_type + " " + std::to_string(port) + " stream";
    start_main = std::chrono::steady_clock::now();
    start_fps = start_main;
}

bool StreamViewer::reconnect() {
    cap.release();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    try {
        cap = open_cap(pipeline);
        return true;
    } catch (...) {
        std::cerr << "reconexion fallida\n";
        return false;
    }
}

void StreamViewer::print_stats() {
    if (frames % fps_interval == 0) {
        auto now = std::chrono::steady_clock::now();
        double fps = fps_interval / std::chrono::duration<double>(now - start_fps).count();
        start_fps = now;
        std::cout << "frames: " << frames
                  << " | fps: " << int(fps)
                  << " | perdidos: " << lost << "\n";
    }
}

void StreamViewer::print_final_stats() {
    auto end_main = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_main - start_main).count();
    
    std::cout << "\n=== estadisticas finales ===\n";
    std::cout << "duracion total: " << duration << " s\n";
    std::cout << "frames totales: " << frames << " | frames perdidos: " << lost << "\n";
    std::cout << "fps promedio: " << (frames / duration) << "\n";
}

void StreamViewer::run() {
    try {
        cap = open_cap(pipeline);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return;
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    while (true) {
        if (!cap.read(frame)) {
            lost++;
            std::cerr << "frame perdido. reconectando...\n";
            if (!reconnect()) break;
            continue;
        }

        frames++;
        cv::resize(frame, display, display_size);
        cv::imshow(window_name, display);

        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;

        print_stats();
    }

    cap.release();
    cv::destroyAllWindows();
    print_final_stats();
}