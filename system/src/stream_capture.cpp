#include "stream_capture.hpp"
#include "utils.hpp"
#include "config.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <iomanip>
#include <sstream>
#include <filesystem>

StreamCapture::StreamCapture(const std::string& user, const std::string& pass,
                             const std::string& ip, int port, const std::string& stream_type)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      frames(0), lost(0), current_fps(0.0), recording_enabled(false),
      stop_signal(nullptr), max_duration(0), fps_interval(Config::DEFAULT_FPS_INTERVAL) {
    
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    start_time = std::chrono::steady_clock::now();
    start_fps = start_time;
    cached_stats = {0.0, 0, 0, cv::Size(0, 0)};
}

StreamCapture::~StreamCapture() {
    release();
}

void StreamCapture::enable_recording(const std::string& output_dir) {
    recording_enabled = true;
    std::filesystem::create_directories(output_dir);
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << output_dir << "/recording_" << stream_type << "_"
       << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".mp4";
    
    output_filename = ss.str();
}

void StreamCapture::set_max_duration(int seconds) {
    max_duration = seconds;
}

void StreamCapture::set_stop_signal(std::atomic<bool>* signal) {
    stop_signal = signal;
}

void StreamCapture::set_fps_interval(int interval) {
    fps_interval = interval;
}

bool StreamCapture::init_recording() {
    if (!recording_enabled || !cap.isOpened()) return false;
    
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    if (fps <= 0) fps = Config::DEFAULT_VIDEO_FPS;
    
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(output_filename, fourcc, fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        spdlog::error("no se pudo crear archivo: {}", output_filename);
        return false;
    }
    
    spdlog::info("grabando {} en: {}", stream_type, output_filename);
    return true;
}

void StreamCapture::stop_recording() {
    if (writer.isOpened()) {
        writer.release();
        spdlog::info("grabacion finalizada: {}", output_filename);
    }
}

bool StreamCapture::open() {
    try {
        cap = open_cap(pipeline);
        if (recording_enabled) {
            return init_recording();
        }
        return true;
    } catch (const std::exception& e) {
        spdlog::error("error al abrir stream {}: {}", stream_type, e.what());
        return false;
    }
}

bool StreamCapture::reconnect() {
    bool was_recording = writer.isOpened();
    if (was_recording) writer.release();
    
    cap.release();
    std::this_thread::sleep_for(std::chrono::seconds(Config::RECONNECT_DELAY_SEC));
    
    try {
        cap = open_cap(pipeline);
        if (was_recording) init_recording();
        return true;
    } catch (...) {
        spdlog::error("reconexion fallida para {}", stream_type);
        return false;
    }
}

bool StreamCapture::read(cv::Mat& frame) {
    if (stop_signal && *stop_signal) return false;
    
    if (max_duration > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= max_duration) {
            return false;
        }
    }
    
    if (!cap.read(frame)) {
        lost++;
        if (!reconnect()) return false;
        return cap.read(frame);
    }
    
    frames++;
    
    if (is_recording()) {
        writer.write(frame);
    }
    
    update_fps();
    return true;
}

void StreamCapture::update_fps() {
    if (frames % fps_interval == 0 && frames > 0) {
        auto now = std::chrono::steady_clock::now();
        current_fps = fps_interval / std::chrono::duration<double>(now - start_fps).count();
        start_fps = now;
    }
    
    if (cap.isOpened()) {
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cached_stats.resolution = cv::Size(width, height);
    }
    
    cached_stats.fps = current_fps;
    cached_stats.frames = frames;
    cached_stats.lost = lost;
}

const StreamStats& StreamCapture::get_stats() {
    return cached_stats;
}

void StreamCapture::print_stats() const {
    if (frames % fps_interval == 0 && frames > 0) {
        std::string rec_status = is_recording() ? " [REC]" : "";
        spdlog::info("{} | frames: {} | fps: {} | perdidos: {}{}", 
                     stream_type, frames, int(current_fps), lost, rec_status);
    }
}

void StreamCapture::print_final_stats() const {
    auto end_time = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    spdlog::info("=== stats {} ===", stream_type);
    spdlog::info("duracion: {:.2f} s", duration);
    spdlog::info("frames: {} | perdidos: {}", frames, lost);
    spdlog::info("fps promedio: {:.2f}", frames / duration);
    
    if (recording_enabled) {
        spdlog::info("archivo: {}", output_filename);
    }
}

void StreamCapture::release() {
    stop_recording();
    if (cap.isOpened()) {
        cap.release();
    }
}