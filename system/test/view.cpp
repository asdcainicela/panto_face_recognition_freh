#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <thread>
#include <vector>

std::atomic<bool> stop_signal(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("cerrando visualizacion...");
        stop_signal = true;
    }
}

void view_stream(StreamCapture& capture, const std::string& stream_type, cv::Point window_pos) {
    if (!capture.open()) {
        spdlog::error("error al abrir stream {}", stream_type);
        return;
    }
    
    std::string window_name = "view - " + stream_type;
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 640, 480);
    cv::moveWindow(window_name, window_pos.x, window_pos.y);
    
    cv::Mat frame, display;
    while (capture.read(frame)) {
        cv::resize(frame, display, cv::Size(640, 480));
        
        const auto& stats = capture.get_stats();
        DrawUtils::DrawConfig config;
        DrawUtils::draw_stream_info(display, stats, stream_type, config);
        
        cv::imshow(window_name, display);
        
        if (cv::waitKey(1) == 27) {
            stop_signal = true;
            break;
        }
        
        capture.print_stats();
    }
    
    cv::destroyWindow(window_name);
    capture.print_final_stats();
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::info);
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::string user = "admin";
    std::string pass = "Panto2025";
    std::string ip = "192.168.0.101";
    int port = 554;
    
    std::vector<std::string> streams;
    
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "main" || arg == "sub") {
            streams.push_back(arg);
        } else if (arg == "both") {
            streams = {"main", "sub"};
        } else {
            streams = {"main", "sub"};
        }
    } else {
        streams = {"main", "sub"};
    }
    
    spdlog::info("visualizando streams...");
    for (const auto& s : streams) {
        spdlog::info("  - {}", s);
    }
    
    std::vector<StreamCapture*> captures;
    std::vector<std::thread> threads;
    
    for (size_t i = 0; i < streams.size(); ++i) {
        auto* cap = new StreamCapture(user, pass, ip, port, streams[i]);
        cap->set_stop_signal(&stop_signal);
        captures.push_back(cap);
        
        cv::Point pos(i * 650, 0);
        threads.emplace_back(view_stream, std::ref(*cap), streams[i], pos);
    }
    
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    for (auto* cap : captures) {
        delete cap;
    }
    
    spdlog::info("visualizacion finalizada");
    return 0;
}