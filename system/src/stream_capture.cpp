#include "stream_capture.hpp"
#include "draw_utils.hpp"
#include <spdlog/spdlog.h>
#include "utils.hpp"        
#include <thread>
#include <iomanip>
#include <sstream>
#include <filesystem>

StreamCapture::StreamCapture(const std::string& user, const std::string& pass,
                             const std::string& ip, int port, const std::string& stream_type)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      frames(0), lost(0), last_frame_count(0), latency_warnings(0), 
      reconnect_count(0), frames_recorded(0),
      current_fps(0.0), recording_enabled(false), recording_paused(false),
      viewing_enabled(false), use_adaptive_latency(true), display_size(640, 480),
      stop_signal(nullptr), max_duration(0), fps_interval(30) {
    
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    window_name = "rtsp " + ip + "/" + stream_type + " " + std::to_string(port) + " stream";
    start_time = std::chrono::steady_clock::now();
    start_fps = start_time;
    last_health_check = start_time;
    cached_stats = {0.0, 0, 0, cv::Size(0, 0), 0, 0};
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

void StreamCapture::enable_viewing(const cv::Size& size) {
    viewing_enabled = true;
    display_size = size;
}

void StreamCapture::enable_adaptive_latency(bool enable) {
    use_adaptive_latency = enable;
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
    
    if (fps <= 0) fps = 25.0;
    
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    writer.open(output_filename, fourcc, fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        spdlog::error("no se pudo crear archivo de grabación: {}", output_filename);
        return false;
    }
    
    recording_paused = false;
    spdlog::info("grabando stream {} en: {}", stream_type, output_filename);
    return true;
}

void StreamCapture::pause_recording() {
    if (writer.isOpened() && !recording_paused) {
        recording_paused = true;
        spdlog::warn("⏸ grabación pausada temporalmente durante reconexión");
    }
}

void StreamCapture::resume_recording() {
    if (recording_paused) {
        recording_paused = false;
        spdlog::info("▶ grabación reanudada");
    }
}

void StreamCapture::stop_recording() {
    if (writer.isOpened()) {
        writer.release();
        spdlog::info("grabación finalizada: {} ({} frames escritos)", output_filename, frames_recorded);
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
        spdlog::error("error abriendo stream: {}", e.what());
        return false;
    }
}

bool StreamCapture::reconnect() {
    reconnect_count++;
    spdlog::warn("🔄 reconectando stream {} (intento #{})", stream_type, reconnect_count);
    
    bool was_recording = writer.isOpened();
    if (was_recording) {
        pause_recording();
    }
    
    cap.release();
    
    int wait_time = std::min(1 + (reconnect_count / 3), 4);
    spdlog::info("esperando {}s antes de reconectar...", wait_time);
    std::this_thread::sleep_for(std::chrono::seconds(wait_time));
    
    try {
        cap = open_cap(pipeline, 3);
        
        if (!verify_stream_health(cap)) {
            spdlog::error("stream reconectado pero no está saludable");
            return false;
        }
        
        if (was_recording && writer.isOpened()) {
            resume_recording();
        }
        
        spdlog::info("✓ reconexión exitosa para stream {}", stream_type);
        
        if (reconnect_count > 0) reconnect_count--;
        
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("reconexión fallida para stream {}: {}", stream_type, e.what());
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
    
    if (is_recording() && !recording_paused && writer.isOpened()) {
        try {
            writer.write(frame);
            frames_recorded++;
        } catch (const std::exception& e) {
            spdlog::warn("error escribiendo frame (no crítico): {}", e.what());
        }
    }
    
    update_fps();
    return true;
}

void StreamCapture::update_fps() {
    auto now = std::chrono::steady_clock::now();
    
    int current_interval = 25;
    bool should_calculate = (frames % current_interval == 0 && frames > 0);
    
    auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(now - start_fps).count();
    if (time_since_last >= 2 && frames > 0) {
        should_calculate = true;
    }
    
    if (should_calculate) {
        double elapsed = std::chrono::duration<double>(now - start_fps).count();
        
        if (elapsed < 0.1) elapsed = 0.1;
        
        double new_fps;
        
        if (frames <= 25) {
            auto total_time = std::chrono::duration<double>(now - start_time).count();
            if (total_time < 0.1) total_time = 0.1;
            new_fps = frames / total_time;
            
            if (new_fps > 30.0) new_fps = 30.0;
        } else {
            int frames_since_last = frames - last_frame_count;
            
            if (last_frame_count == 0) {
                frames_since_last = frames;
                elapsed = std::chrono::duration<double>(now - start_time).count();
            }
            
            if (frames_since_last <= 0) frames_since_last = 1;
            if (elapsed < 0.1) elapsed = 0.1;
            
            new_fps = frames_since_last / elapsed;
            
            if (new_fps > 30.0) new_fps = 30.0;
            if (new_fps < 0.5) new_fps = 0.5;
        }
        
        last_frame_count = frames;
        
        if (current_fps > 0 && frames > 50) {
            new_fps = current_fps * 0.7 + new_fps * 0.3;
        }
        
        bool fps_changed_significantly = false;
        if (current_fps > 0 && frames > 25) {
            double fps_change_ratio = std::abs(new_fps - current_fps) / current_fps;
            if (fps_change_ratio > 0.3) {
                fps_changed_significantly = true;
                spdlog::info("cambio significativo de FPS en {}: {:.1f} -> {:.1f}", 
                            stream_type, current_fps, new_fps);
            }
        }
        
        current_fps = new_fps;
        start_fps = now;
        
        if (use_adaptive_latency && fps_changed_significantly && frames > 50) {
            auto since_last_update = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_health_check).count();
            
            if (since_last_update >= 20) {
                spdlog::info("ajustando latencia adaptativa (fps actual: {:.1f})", current_fps);
                
                std::string new_pipeline = gst_pipeline_adaptive(user, pass, ip, port, stream_type, current_fps);
                
                if (new_pipeline != pipeline) {
                    pipeline = new_pipeline;
                    spdlog::debug("pipeline actualizado para optimizar latencia");
                }
                
                last_health_check = now;
            }
        }
    }
    
    if (cap.isOpened()) {
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cached_stats.resolution = cv::Size(width, height);
    }
    
    cached_stats.fps = current_fps;
    cached_stats.frames = frames;
    cached_stats.lost = lost;
    cached_stats.reconnects = reconnect_count;
    cached_stats.latency_warnings = latency_warnings;
}

void StreamCapture::run() {
    if (viewing_enabled) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, display_size.width, display_size.height);
        
        if (stream_type == "main") {
            cv::moveWindow(window_name, 0, 0);
        } else {
            cv::moveWindow(window_name, 800, 0);
        }
        
        cv::Mat splash = cv::Mat::zeros(display_size, CV_8UC3);
        cv::putText(splash, "Conectando a " + stream_type + "...", 
                    cv::Point(50, display_size.height/2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::imshow(window_name, splash);
        cv::waitKey(1);
    }
    
    if (!open()) {
        spdlog::error("error al abrir stream {}", stream_type);
        if (viewing_enabled) {
            cv::destroyWindow(window_name);
        }
        return;
    }
    
    cv::Mat frame, display;
    int consecutive_fails = 0;
    const int MAX_CONSECUTIVE_FAILS = 100;
    const int RECONNECT_THRESHOLD = 5;
    
    int frames_since_flush = 0;
    const int FLUSH_INTERVAL = (stream_type == "main") ? 10 : 30;
    
    while (true) {
        if (stop_signal && *stop_signal) break;
        
        if (max_duration > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= max_duration) {
                spdlog::info("⏱ duración objetivo alcanzada: {}s", max_duration);
                break;
            }
        }
        
        if (!cap.read(frame)) {
            lost++;
            consecutive_fails++;
            
            if (consecutive_fails >= RECONNECT_THRESHOLD) {
                spdlog::warn("⚠️ {} fallos consecutivos, intentando reconectar...", consecutive_fails);
                
                if (!reconnect()) {
                    if (consecutive_fails >= MAX_CONSECUTIVE_FAILS) {
                        spdlog::error("❌ demasiados fallos ({}), abortando stream {}", 
                                     consecutive_fails, stream_type);
                        break;
                    }
                } else {
                    consecutive_fails = 0;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        consecutive_fails = 0;
        frames++;
        frames_since_flush++;
        
        if (frames_since_flush >= FLUSH_INTERVAL) {
            cv::Mat dummy;
            int flushed = 0;
            while (flushed < 5 && cap.grab()) {
                flushed++;
            }
            if (flushed > 0) {
                latency_warnings++;
                if (stream_type == "main") {
                    spdlog::warn("🗑️ {} flushed {} frames acumulados (total warnings: {})", 
                                stream_type, flushed, latency_warnings);
                }
            } else {
                if (latency_warnings > 0) latency_warnings--;
            }
            frames_since_flush = 0;
        }
        
        if (is_recording() && !recording_paused && writer.isOpened()) {
            try {
                writer.write(frame);
                frames_recorded++;
            } catch (const std::exception& e) {
                spdlog::warn("error escribiendo frame (no crítico): {}", e.what());
            }
        }
        
        if (viewing_enabled) {
            cv::resize(frame, display, display_size);
            update_fps();
            
            const auto& s = get_stats();
            cv::Scalar color = is_recording() ? 
                (recording_paused ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255)) : 
                cv::Scalar(255, 0, 0);
            
            if (frames <= 50) {
                color = cv::Scalar(0, 255, 255);
            }
            
            cv::rectangle(display, cv::Rect(0, 0, display.cols/4 + display.cols/16, display.rows/4 + 60), color, 0.5);
            
            cv::putText(display, "channel: " + stream_type, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            cv::putText(display, "frames: " + std::to_string(s.frames), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            
            std::string fps_text;
            if (frames <= 50 || current_fps <= 15) {
                fps_text = "fps: " + std::to_string(static_cast<int>(s.fps * 10) / 10.0);
            } else {
                fps_text = "fps: " + std::to_string(static_cast<int>(s.fps));
            }
            cv::putText(display, fps_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            
            cv::putText(display, "lost: " + std::to_string(s.lost), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            cv::putText(display, "resolution: " + std::to_string(frame.cols) + "*" + std::to_string(frame.rows), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            cv::putText(display, "reconnects: " + std::to_string(reconnect_count), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            
            if (stream_type == "main" && latency_warnings > 0) {
                cv::putText(display, "LATENCY: " + std::to_string(latency_warnings), 
                           cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
            }
            
            if (is_recording()) {
                std::string rec_text = recording_paused ? "REC-PAUSED" : "REC";
                cv::putText(display, rec_text, cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                           recording_paused ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255), 2);
                cv::putText(display, "recorded: " + std::to_string(frames_recorded), cv::Point(10, 160), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
            
            if (use_adaptive_latency) {
                cv::putText(display, "ADAPTIVE", cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
            }
            
            if (frames <= 50) {
                cv::putText(display, "STARTING...", cv::Point(display.cols - 150, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }
            
            cv::imshow(window_name, display);
            
            if (cv::waitKey(1) == 27) {
                if (stop_signal) *stop_signal = true;
                break;
            }
        }
        
        if (frames % 25 == 0 && frames > 0) {
            print_stats();
        }
    }
    
    if (viewing_enabled) {
        cv::destroyWindow(window_name);
    }
    
    print_final_stats();
}

const StreamStats& StreamCapture::get_stats() {
    return cached_stats;
}

void StreamCapture::print_stats() const {
    if (frames % 25 == 0 && frames > 0) {
        const auto& s = cached_stats;
        std::string rec_status = is_recording() ? 
            (recording_paused ? " [REC-PAUSED]" : " [REC]") : "";
        std::string adaptive = use_adaptive_latency ? " [ADAPTIVE]" : "";
        
        std::string fps_status;
        if (current_fps <= 5) fps_status = " ⚠️";
        else if (current_fps <= 10) fps_status = " ⚡";
        else if (current_fps >= 20) fps_status = " ✓";
        else fps_status = "";
        
        std::string phase = frames <= 50 ? " [STARTING]" : "";
        
        spdlog::info("stream: {} | frames: {} | fps: {:.1f}{} | perdidos: {} | rec: {} | reconex: {}{}{}{}", 
                     stream_type, s.frames, s.fps, fps_status, s.lost, 
                     frames_recorded, reconnect_count, rec_status, adaptive, phase);
    }
}

void StreamCapture::print_final_stats() const {
    auto end_time = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    spdlog::info("=== estadisticas finales {} ===", stream_type);
    spdlog::info("duracion total: {:.2f} s", duration);
    spdlog::info("frames totales: {} | frames perdidos: {}", frames, lost);
    spdlog::info("fps promedio: {:.2f}", frames / duration);
    spdlog::info("reconexiones totales: {}", reconnect_count);
    
    if (recording_enabled) {
        spdlog::info("archivo grabado: {} ({} frames)", output_filename, frames_recorded);
        double recording_efficiency = frames > 0 ? (frames_recorded * 100.0 / frames) : 0.0;
        spdlog::info("eficiencia grabación: {:.1f}%", recording_efficiency);
    }
}

void StreamCapture::release() {
    stop_recording();
    if (cap.isOpened()) {
        cap.release();
    }
}