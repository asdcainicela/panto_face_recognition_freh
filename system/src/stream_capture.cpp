// ============= src/stream_capture.cpp =============
#include "stream_capture.hpp"
#include "utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>

// ==================== CAPTURE THREAD ====================

CaptureThread::CaptureThread(const std::string& pipeline, FrameQueue& queue)
    : pipeline(pipeline), running(false), frame_queue(queue),
      frames_captured(0), frames_lost(0), reconnects(0), consecutive_fails(0)
{
}

CaptureThread::~CaptureThread() {
    stop();
}

bool CaptureThread::open_capture(int retries) {
    for (int attempt = 0; attempt < retries; attempt++) {
        spdlog::info("🔌 [Capture Thread] Intento {}/{} conectar...", attempt + 1, retries);
        
        try {
            cap = open_cap(pipeline, 1);  // Un solo intento interno
            
            if (cap.isOpened()) {
                // Verificar que puede leer
                cv::Mat test_frame;
                for (int i = 0; i < 3; i++) {
                    if (cap.read(test_frame) && !test_frame.empty()) {
                        spdlog::info("✓ [Capture Thread] Conexión OK - {}x{}", 
                                   test_frame.cols, test_frame.rows);
                        return true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                }
                
                spdlog::warn("⚠️ [Capture Thread] No se pudo leer frame de prueba");
                cap.release();
            }
        } catch (const std::exception& e) {
            spdlog::error("✗ [Capture Thread] Excepción: {}", e.what());
        }
        
        if (attempt < retries - 1) {
            int wait_time = 2 + attempt;  // Backoff: 2s, 3s, 4s...
            spdlog::info("⏳ [Capture Thread] Esperando {}s antes de reintentar...", wait_time);
            std::this_thread::sleep_for(std::chrono::seconds(wait_time));
        }
    }
    
    return false;
}

bool CaptureThread::reconnect() {
    reconnects++;
    spdlog::warn("🔄 [Capture Thread] Reconectando... (intento #{})", reconnects.load());
    
    if (cap.isOpened()) {
        cap.release();
    }
    
    // Backoff exponencial
    int wait_time = std::min(1 + (reconnects.load() / 3), 4);
    spdlog::info("⏳ [Capture Thread] Esperando {}s antes de reconectar...", wait_time);
    std::this_thread::sleep_for(std::chrono::seconds(wait_time));
    
    if (open_capture(3)) {
        consecutive_fails = 0;
        spdlog::info("✓ [Capture Thread] Reconexión exitosa");
        return true;
    }
    
    spdlog::error("❌ [Capture Thread] Reconexión fallida");
    return false;
}

void CaptureThread::capture_loop() {
    spdlog::info("🚀 [Capture Thread] Iniciando loop de captura...");
    
    // Abrir captura inicial
    if (!open_capture(5)) {
        spdlog::error("❌ [Capture Thread] No se pudo abrir captura inicial");
        return;
    }
    
    int error_count = 0;
    int frames_since_health_check = 0;
    
    while (running.load()) {
        cv::Mat frame;
        bool ok = cap.read(frame);
        
        if (!ok || frame.empty() || frame.total() == 0) {
            frames_lost++;
            consecutive_fails++;
            error_count++;
            
            // ¿Necesitamos reconectar?
            if (consecutive_fails >= reconnect_threshold) {
                spdlog::warn("⚠️ [Capture Thread] {} fallos consecutivos, reconectando...", 
                           consecutive_fails.load());
                
                if (!reconnect()) {
                    if (consecutive_fails >= max_consecutive_fails) {
                        spdlog::error("❌ [Capture Thread] Demasiados fallos ({}), abortando", 
                                    consecutive_fails.load());
                        break;
                    }
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Frame válido
        consecutive_fails = 0;
        error_count = 0;
        frames_captured++;
        frames_since_health_check++;
        
        // Enviar a queue (thread-safe)
        frame_queue.push(frame);
        
        // Health check periódico (cada 100 frames)
        if (frames_since_health_check >= 100) {
            if (!cap.isOpened()) {
                spdlog::warn("⚠️ [Capture Thread] Health check falló");
                reconnect();
            }
            frames_since_health_check = 0;
        }
        
        // Log cada 30 frames
        if (frames_captured % 30 == 0) {
            spdlog::info("📊 [Capture Thread] Frames: {} | Lost: {} | Queue: {} | Reconnects: {}",
                       frames_captured.load(), frames_lost.load(), 
                       frame_queue.size(), reconnects.load());
        }
        
        // Pequeña pausa para no saturar
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    if (cap.isOpened()) {
        cap.release();
    }
    
    spdlog::info("🛑 [Capture Thread] Loop finalizado");
}

void CaptureThread::start() {
    if (running.load()) {
        spdlog::warn("⚠️ [Capture Thread] Ya está corriendo");
        return;
    }
    
    running = true;
    thread = std::thread(&CaptureThread::capture_loop, this);
}

void CaptureThread::stop() {
    if (!running.load()) return;
    
    spdlog::info("🛑 [Capture Thread] Deteniendo...");
    running = false;
    
    if (thread.joinable()) {
        thread.join();
    }
}

// ==================== STREAM CAPTURE ====================

StreamCapture::StreamCapture(const std::string& user, const std::string& pass,
                             const std::string& ip, int port, 
                             const std::string& stream_type)
    : user(user), pass(pass), ip(ip), port(port), stream_type(stream_type),
      frame_queue(3),  // Max 3 frames en queue
      frames_displayed(0), current_fps(0.0),
      viewing_enabled(false), display_size(640, 480),
      stop_signal(nullptr), fps_interval(30)
{
    pipeline = gst_pipeline(user, pass, ip, port, stream_type);
    window_name = "RTSP " + ip + "/" + stream_type;
    
    start_time = std::chrono::steady_clock::now();
    start_fps = start_time;
    
    cached_stats = {0.0, 0, 0, cv::Size(0, 0), 0, 0};
}

StreamCapture::~StreamCapture() {
    release();
}

void StreamCapture::enable_viewing(const cv::Size& size) {
    viewing_enabled = true;
    display_size = size;
}

void StreamCapture::set_stop_signal(std::atomic<bool>* signal) {
    stop_signal = signal;
}

void StreamCapture::set_fps_interval(int interval) {
    fps_interval = interval;
}

bool StreamCapture::open() {
    capture_thread = std::make_unique<CaptureThread>(pipeline, frame_queue);
    capture_thread->start();
    
    // Esperar primer frame
    cv::Mat test_frame;
    if (!frame_queue.pop(test_frame, 5000)) {  // 5s timeout
        spdlog::error("❌ No se recibió frame inicial");
        return false;
    }
    
    spdlog::info("✓ Stream abierto y recibiendo frames");
    return true;
}

bool StreamCapture::read(cv::Mat& frame) {
    return frame_queue.pop(frame, 1000);  // 1s timeout
}

void StreamCapture::release() {
    if (capture_thread) {
        capture_thread->stop();
        capture_thread.reset();
    }
    frame_queue.clear();
}

void StreamCapture::run() {
    if (viewing_enabled) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, display_size.width, display_size.height);
    }
    
    if (!open()) {
        spdlog::error("❌ Error al abrir stream");
        return;
    }
    
    cv::Mat frame, display;
    
    while (true) {
        if (stop_signal && stop_signal->load()) break;
        
        // Obtener frame del thread de captura
        if (!frame_queue.pop(frame, 1000)) {
            // Timeout - verificar si el thread sigue vivo
            if (!capture_thread->is_running()) {
                spdlog::error("❌ Thread de captura murió");
                break;
            }
            continue;
        }
        
        frames_displayed++;
        update_fps();
        
        if (viewing_enabled) {
            cv::resize(frame, display, display_size);
            
            // Overlay de información
            int y = 20;
            cv::putText(display, "Stream: " + stream_type, cv::Point(10, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            y += 20;
            
            cv::putText(display, "FPS: " + std::to_string(static_cast<int>(current_fps)),
                       cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 0, 0), 1);
            y += 20;
            
            cv::putText(display, "Frames: " + std::to_string(capture_thread->get_frames()),
                       cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 0, 0), 1);
            y += 20;
            
            cv::putText(display, "Lost: " + std::to_string(capture_thread->get_lost()),
                       cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 0, 0), 1);
            y += 20;
            
            cv::putText(display, "Reconnects: " + std::to_string(capture_thread->get_reconnects()),
                       cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 0, 0), 1);
            y += 20;
            
            cv::putText(display, "Queue: " + std::to_string(frame_queue.size()),
                       cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 0, 0), 1);
            
            cv::imshow(window_name, display);
            
            if (cv::waitKey(1) == 27) {  // ESC
                if (stop_signal) stop_signal->store(true);
                break;
            }
        }
        
        if (frames_displayed % fps_interval == 0) {
            print_stats();
        }
    }
    
    if (viewing_enabled) {
        cv::destroyWindow(window_name);
    }
    
    print_final_stats();
}

void StreamCapture::update_fps() {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_fps).count();
    
    if (elapsed >= 1.0) {  // Actualizar cada segundo
        current_fps = frames_displayed / 
                     std::chrono::duration<double>(now - start_time).count();
        start_fps = now;
    }
    
    cached_stats.fps = current_fps;
    cached_stats.frames = capture_thread->get_frames();
    cached_stats.lost = capture_thread->get_lost();
    cached_stats.reconnects = capture_thread->get_reconnects();
}

const StreamStats& StreamCapture::get_stats() {
    return cached_stats;
}

void StreamCapture::print_stats() const {
    spdlog::info("Stream: {} | Frames: {} | FPS: {:.1f} | Lost: {} | Reconnects: {}",
                stream_type, capture_thread->get_frames(), current_fps,
                capture_thread->get_lost(), capture_thread->get_reconnects());
}

void StreamCapture::print_final_stats() const {
    auto end_time = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    
    spdlog::info("=== Estadísticas finales {} ===", stream_type);
    spdlog::info("Duración total: {:.2f}s", duration);
    spdlog::info("Frames capturados: {}", capture_thread->get_frames());
    spdlog::info("Frames perdidos: {}", capture_thread->get_lost());
    spdlog::info("FPS promedio: {:.2f}", capture_thread->get_frames() / duration);
    spdlog::info("Reconexiones totales: {}", capture_thread->get_reconnects());
}