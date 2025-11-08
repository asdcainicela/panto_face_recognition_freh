#include "stream_viewer.hpp"
#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>

StreamViewer::StreamViewer(const std::string& user, const std::string& pass, 
                           const std::string& ip, int port, const std::string& stream_type,
                           cv::Size display_size, int fps_interval)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      display_size(display_size), fps_interval(fps_interval),
      frames(0), lost(0), current_fps(0.0) {
    
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    window_name = "rtsp " + ip + "/" + stream_type + " " + std::to_string(port) + " stream";
    start_main = std::chrono::steady_clock::now();
    start_fps = start_main;
    cached_stats = {0.0, 0, 0};
}

bool StreamViewer::reconnect() {
    cap.release();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    try {
        cap = open_cap(pipeline);
        return true;
    } catch (...) {
        spdlog::error("reconexion fallida para stream {}", stream_type);
        return false;
    }
}

void StreamViewer::update_fps() {
    if (frames % fps_interval == 0 && frames > 0) {
        auto now = std::chrono::steady_clock::now();
        current_fps = fps_interval / std::chrono::duration<double>(now - start_fps).count();
        start_fps = now;
    }
    
    cached_stats.fps = current_fps;
    cached_stats.frames = frames;
    cached_stats.lost = lost;
}

const StreamStats& StreamViewer::get_stats() {
    return cached_stats;
}

void StreamViewer::print_stats() {
    if (frames % fps_interval == 0) {
        const auto& s = get_stats();
        spdlog::info("stream: {} | frames: {} | fps: {} | perdidos: {}", 
                     stream_type, s.frames, int(s.fps), s.lost);
    }
}

void StreamViewer::print_final_stats() {
    auto end_main = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_main - start_main).count();
    
    spdlog::info("=== estadisticas finales {} ===", stream_type);
    spdlog::info("duracion total: {:.2f} s", duration);
    spdlog::info("frames totales: {} | frames perdidos: {}", frames, lost);
    spdlog::info("fps promedio: {:.2f}", frames / duration);
}

void StreamViewer::run() {
    try {
        cap = open_cap(pipeline);
    } catch (const std::exception& e) {
        spdlog::error("error al abrir stream {}: {}", stream_type, e.what());
        return;
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, display_size.width, display_size.height);
    
    if (stream_type == "main") {
        cv::moveWindow(window_name, 0, 0);
    } else {
        cv::moveWindow(window_name, 800, 0);
    }

    while (true) {
        if (!cap.read(frame)) {
            lost++;
            spdlog::warn("frame perdido en {}. reconectando...", stream_type);
            if (!reconnect()) break;
            continue;
        }

        frames++;

        auto w_frame = frame.cols;
        auto h_frame = frame.rows;

        cv::resize(frame, display, display_size);
        update_fps();

        int w = display.cols;
        int h = display.rows;
        
        cv::Rect roi(0, 0, w/4 + w/16, h/4);
        cv::Scalar color_text(255, 0, 0);
        
        cv::rectangle(display, roi, color_text, 0.5);
        
        const auto& s = get_stats();
        
        std::string text_channel = "channel: " + stream_type;
        std::string text_frame = "frames: " + std::to_string(s.frames);
        std::string text_fps = "fps: " + std::to_string(int(s.fps));
        std::string text_perdidos = "lost: " + std::to_string(s.lost);
        std::string text_resolution = "resolution: " + std::to_string(w_frame) + "*" + std::to_string(h_frame);

        cv::putText(display, text_channel, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1);
        cv::putText(display, text_frame, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1);
        cv::putText(display, text_fps, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1);
        cv::putText(display, text_perdidos, cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1);
        cv::putText(display, text_resolution, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1);

        cv::imshow(window_name, display);

        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;

        print_stats();
    }

    cap.release();
    cv::destroyAllWindows();
    print_final_stats();
}