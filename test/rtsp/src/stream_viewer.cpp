#include "stream_viewer.hpp"
#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <iomanip>
#include <sstream>
#include <filesystem>

StreamViewer::StreamViewer(const std::string& user, const std::string& pass, 
                           const std::string& ip, int port, const std::string& stream_type,
                           cv::Size display_size, int fps_interval)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      display_size(display_size), fps_interval(fps_interval),
      frames(0), lost(0), current_fps(0.0), recording_enabled(false),
      stop_signal(nullptr), max_duration(0), reconnect_count(0), 
      use_adaptive_latency(true) {  // Activar latencia adaptativa por defecto
    
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    window_name = "rtsp " + ip + "/" + stream_type + " " + std::to_string(port) + " stream";
    start_main = std::chrono::steady_clock::now();
    start_fps = start_main;
    last_health_check = start_main;
    cached_stats = {0.0, 0, 0};
}

void StreamViewer::enable_recording(const std::string& output_dir) {
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

void StreamViewer::set_max_duration(int seconds) {
    max_duration = seconds;
}

void StreamViewer::set_stop_signal(std::atomic<bool>* signal) {
    stop_signal = signal;
}

void StreamViewer::enable_adaptive_latency(bool enable) {
    use_adaptive_latency = enable;
}

bool StreamViewer::init_recording() {
    if (!recording_enabled) return true;
    
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    if (fps <= 0) fps = 25.0;
    
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer.open(output_filename, fourcc, fps, cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        spdlog::error("no se pudo crear archivo de grabación: {}", output_filename);
        return false;
    }
    
    spdlog::info("grabando stream {} en: {}", stream_type, output_filename);
    return true;
}

void StreamViewer::stop_recording() {
    if (writer.isOpened()) {
        writer.release();
        spdlog::info("grabación finalizada: {}", output_filename);
    }
}

bool StreamViewer::reconnect() {
    reconnect_count++;
    spdlog::warn("reconectando stream {} (intento #{})", stream_type, reconnect_count);
    
    bool was_recording = writer.isOpened();
    if (was_recording) writer.release();
    
    cap.release();
    
    // Espera progresiva: 1s, 2s, 3s, hasta máximo 5s
    int wait_time = std::min(reconnect_count, 5);
    spdlog::info("esperando {}s antes de reconectar...", wait_time);
    std::this_thread::sleep_for(std::chrono::seconds(wait_time));
    
    try {
        // Usar latencia adaptativa en reconexión si está activada
        if (use_adaptive_latency && current_fps > 0) {
            pipeline = gst_pipeline_adaptive(user, pass, ip, port, stream_type, current_fps);
            spdlog::info("usando pipeline adaptativo para reconexión");
        }
        
        cap = open_cap(pipeline, 3);  // Menos reintentos en reconexión
        
        // Verificar salud del stream
        if (!verify_stream_health(cap)) {
            spdlog::error("stream reconectado pero no está saludable");
            return false;
        }
        
        if (was_recording) init_recording();
        
        spdlog::info("✓ reconexión exitosa para stream {}", stream_type);
        reconnect_count = 0;  // Reset contador en reconexión exitosa
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("reconexión fallida para stream {}: {}", stream_type, e.what());
        return false;
    }
}

void StreamViewer::update_fps() {
    if (frames % fps_interval == 0 && frames > 0) {
        auto now = std::chrono::steady_clock::now();
        double new_fps = fps_interval / std::chrono::duration<double>(now - start_fps).count();
        
        // Detectar cambios bruscos de FPS
        bool fps_changed_significantly = false;
        if (current_fps > 0) {
            double fps_change_ratio = std::abs(new_fps - current_fps) / current_fps;
            if (fps_change_ratio > 0.3) {  // Cambió más del 30%
                fps_changed_significantly = true;
                spdlog::info("cambio significativo de FPS en {}: {:.1f} -> {:.1f}", 
                            stream_type, current_fps, new_fps);
            }
        }
        
        current_fps = new_fps;
        start_fps = now;
        
        // Actualizar pipeline con latencia adaptativa
        if (use_adaptive_latency) {
            auto since_last_update = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_health_check).count();
            
            // Actualizar cada 30 segundos O si hay cambio brusco de FPS
            if (since_last_update >= 30 || fps_changed_significantly) {
                spdlog::info("ajustando latencia adaptativa (fps actual: {:.1f})", current_fps);
                
                // Regenerar pipeline con nueva latencia
                std::string new_pipeline = gst_pipeline_adaptive(user, pass, ip, port, stream_type, current_fps);
                
                // Solo actualizar si el pipeline cambió significativamente
                if (new_pipeline != pipeline) {
                    pipeline = new_pipeline;
                    spdlog::debug("pipeline actualizado para optimizar latencia");
                }
                
                last_health_check = now;
            }
        }
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
        std::string rec_status = is_recording() ? " [REC]" : "";
        std::string adaptive = use_adaptive_latency ? " [ADAPTIVE]" : "";
        
        // Indicador visual de calidad de FPS
        std::string fps_status;
        if (current_fps <= 5) fps_status = " ⚠️ FPS CRÍTICO";
        else if (current_fps <= 10) fps_status = " ⚡ FPS BAJO";
        else if (current_fps >= 25) fps_status = " ✓ FPS ALTO";
        else fps_status = "";
        
        spdlog::info("stream: {} | frames: {} | fps: {:.1f}{} | perdidos: {} | reconexiones: {}{}{}", 
                     stream_type, s.frames, s.fps, fps_status, s.lost, reconnect_count, rec_status, adaptive);
    }
}

void StreamViewer::print_final_stats() {
    auto end_main = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_main - start_main).count();
    
    spdlog::info("=== estadisticas finales {} ===", stream_type);
    spdlog::info("duracion total: {:.2f} s", duration);
    spdlog::info("frames totales: {} | frames perdidos: {}", frames, lost);
    spdlog::info("fps promedio: {:.2f}", frames / duration);
    spdlog::info("reconexiones totales: {}", reconnect_count);
    
    if (recording_enabled) {
        spdlog::info("archivo grabado: {}", output_filename);
    }
}

void StreamViewer::run() {
    try {
        cap = open_cap(pipeline);
    } catch (const std::exception& e) {
        spdlog::error("error al abrir stream {}: {}", stream_type, e.what());
        return;
    }

    if (recording_enabled && !init_recording()) {
        cap.release();
        return;
    }

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, display_size.width, display_size.height);
    
    if (stream_type == "main") {
        cv::moveWindow(window_name, 0, 0);
    } else {
        cv::moveWindow(window_name, 800, 0);
    }

    int consecutive_fails = 0;
    const int MAX_CONSECUTIVE_FAILS = 50;  // Reintentar reconectar hasta 50 fallos consecutivos

    while (true) {
        if (stop_signal && *stop_signal) break;
        
        if (max_duration > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_main;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= max_duration) break;
        }
        
        if (!cap.read(frame)) {
            lost++;
            consecutive_fails++;
            
            if (consecutive_fails >= MAX_CONSECUTIVE_FAILS) {
                spdlog::error("demasiados fallos consecutivos ({}), abortando stream {}", 
                             consecutive_fails, stream_type);
                break;
            }
            
            // Verificar si el stream sigue vivo antes de reconectar
            if (!verify_stream_health(cap)) {
                if (!reconnect()) {
                    spdlog::error("no se pudo reconectar, finalizando stream {}", stream_type);
                    break;
                }
                consecutive_fails = 0;  // Reset en reconexión exitosa
            }
            continue;
        }

        consecutive_fails = 0;  // Reset en lectura exitosa
        frames++;
        
        if (is_recording()) writer.write(frame);

        cv::resize(frame, display, display_size);
        update_fps();

        const auto& s = get_stats();
        cv::Scalar color = is_recording() ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        
        cv::rectangle(display, cv::Rect(0, 0, display.cols/4 + display.cols/16, display.rows/4 + 40), color, 0.5);
        
        cv::putText(display, "channel: " + stream_type, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(display, "frames: " + std::to_string(s.frames), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(display, "fps: " + std::to_string(int(s.fps)), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(display, "lost: " + std::to_string(s.lost), cv::Point(10, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(display, "resolution: " + std::to_string(frame.cols) + "*" + std::to_string(frame.rows), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        cv::putText(display, "reconnects: " + std::to_string(reconnect_count), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        
        if (is_recording()) {
            cv::putText(display, "REC", cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
        
        if (use_adaptive_latency) {
            cv::putText(display, "ADAPTIVE", cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow(window_name, display);

        if (cv::waitKey(1) == 27) break;

        print_stats();
    }

    stop_recording();
    cap.release();
    cv::destroyAllWindows();
    print_final_stats();
}