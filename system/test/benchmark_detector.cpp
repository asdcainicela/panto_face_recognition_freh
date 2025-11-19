// ============= test/benchmark_detector.cpp =============
#include "detector_optimized.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

struct BenchmarkResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double fps;
    int frames;
    
    // Profiling detallado (solo para optimizado)
    double avg_preprocess_ms = 0;
    double avg_inference_ms = 0;
    double avg_postprocess_ms = 0;
};

template<typename DetectorType>
BenchmarkResult benchmark_detector(DetectorType& detector, 
                                   const cv::Mat& test_img,
                                   int warmup = 10,
                                   int iterations = 100,
                                   bool profile = false) 
{
    // Warmup
    spdlog::info("  Warmup: {} iterations...", warmup);
    for (int i = 0; i < warmup; i++) {
        detector.detect(test_img);
    }
    
    // Benchmark
    spdlog::info("  Running: {} iterations...", iterations);
    std::vector<double> times;
    double total_preproc = 0, total_infer = 0, total_postproc = 0;
    
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto dets = detector.detect(test_img);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        
        // Si es detector optimizado, extraer profile
        if (profile) {
            if constexpr (std::is_same_v<DetectorType, FaceDetectorOptimized>) {
                total_preproc += detector.last_profile.preprocess_ms;
                total_infer += detector.last_profile.inference_ms;
                total_postproc += detector.last_profile.postprocess_ms;
            }
        }
    }
    
    // Stats
    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times) {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    
    BenchmarkResult result;
    result.avg_ms = sum / iterations;
    result.min_ms = min_t;
    result.max_ms = max_t;
    result.fps = 1000.0 / result.avg_ms;
    result.frames = iterations;
    
    if (profile) {
        result.avg_preprocess_ms = total_preproc / iterations;
        result.avg_inference_ms = total_infer / iterations;
        result.avg_postprocess_ms = total_postproc / iterations;
    }
    
    return result;
}

void print_table_header() {
    std::cout << "\n" << std::string(90, '=') << std::endl;
    std::cout << "| Resolución    | Avg (ms) | Min (ms) | Max (ms) |  FPS  | Detecciones |" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

void print_table_row(const cv::Size& res, const BenchmarkResult& r, int det_count) {
    printf("| %4dx%-4d    | %8.2f | %8.2f | %8.2f | %5.1f | %11d |\n",
           res.width, res.height,
           r.avg_ms, r.min_ms, r.max_ms, r.fps, det_count);
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S] %v");
    spdlog::set_level(spdlog::level::info);
    
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <model.engine> <test_image.jpg> [--gpu-preproc]" << std::endl;
        std::cerr << "\nOpciones:" << std::endl;
        std::cerr << "  --gpu-preproc   Usar GPU preprocessing (default: ON)" << std::endl;
        std::cerr << "  --cpu-preproc   Forzar CPU preprocessing" << std::endl;
        return 1;
    }
    
    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    bool gpu_preproc = true;
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--cpu-preproc") gpu_preproc = false;
        if (arg == "--gpu-preproc") gpu_preproc = true;
    }
    
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        spdlog::error("Error leyendo: {}", image_path);
        return 1;
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       BENCHMARK SCRFD TensorRT - PANTO System             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    spdlog::info("Imagen test: {}x{}", img.cols, img.rows);
    spdlog::info("Engine: {}", engine_path);
    spdlog::info("GPU Preprocessing: {}", gpu_preproc ? "ENABLED" : "DISABLED");
    
    try {
        // ========== TEST 1: Detector Original ==========
        spdlog::info("\n🔵 TEST 1: Detector Original (baseline)");
        FaceDetector detector_original(engine_path);
        detector_original.set_conf_threshold(0.5f);
        
        cv::Mat test_1080p;
        cv::resize(img, test_1080p, cv::Size(1920, 1080));
        
        auto baseline = benchmark_detector(detector_original, test_1080p, 10, 50);
        auto baseline_dets = detector_original.detect(test_1080p);
        
        spdlog::info("✓ Baseline: {:.2f}ms ({:.1f} FPS) - {} rostros detectados",
                    baseline.avg_ms, baseline.fps, baseline_dets.size());
        
        // ========== TEST 2: Detector Optimizado ==========
        spdlog::info("\n🚀 TEST 2: Detector Optimizado");
        FaceDetectorOptimized detector_opt(engine_path, gpu_preproc);
        detector_opt.set_conf_threshold(0.5f);
        
        auto optimized = benchmark_detector(detector_opt, test_1080p, 10, 50, true);
        auto optimized_dets = detector_opt.detect(test_1080p);
        
        spdlog::info("✓ Optimizado: {:.2f}ms ({:.1f} FPS) - {} rostros detectados",
                    optimized.avg_ms, optimized.fps, optimized_dets.size());
        
        // Profiling detallado
        if (optimized.avg_preprocess_ms > 0) {
            spdlog::info("   └─ Preprocessing: {:.2f}ms", optimized.avg_preprocess_ms);
            spdlog::info("   └─ Inference:     {:.2f}ms", optimized.avg_inference_ms);
            spdlog::info("   └─ Postprocess:   {:.2f}ms", optimized.avg_postprocess_ms);
        }
        
        // ========== SPEEDUP ==========
        double speedup = baseline.avg_ms / optimized.avg_ms;
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    RESULTADOS FINALES                      ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        printf("  Original:    %6.2f ms  (%5.1f FPS)\n", baseline.avg_ms, baseline.fps);
        printf("  Optimizado:  %6.2f ms  (%5.1f FPS)\n", optimized.avg_ms, optimized.fps);
        printf("  ───────────────────────────────────────\n");
        printf("  Speedup:     %.2fx\n", speedup);
        printf("  Ganancia:    %.1f ms ahorrados por frame\n", baseline.avg_ms - optimized.avg_ms);
        std::cout << "\n";
        
        if (speedup < 1.2) {
            spdlog::warn("⚠️  Speedup bajo. Posibles causas:");
            spdlog::warn("   - OpenCV no compilado con CUDA");
            spdlog::warn("   - Engine no optimizado (FP32 en lugar de FP16)");
            spdlog::warn("   - GPU compartida con otras tareas");
        } else {
            spdlog::info("✓ Optimización exitosa! {:.1f}x más rápido", speedup);
        }
        
        // ========== TEST 3: Múltiples Resoluciones ==========
        spdlog::info("\n📊 TEST 3: Benchmark por Resolución (Optimizado)");
        
        std::vector<cv::Size> resolutions = {
            {640, 480},    // VGA
            {1280, 720},   // HD
            {1920, 1080},  // FHD
            {2560, 1440},  // QHD
        };
        
        if (img.cols >= 3840 && img.rows >= 2160) {
            resolutions.push_back({3840, 2160});  // 4K
        }
        
        print_table_header();
        
        for (const auto& res : resolutions) {
            if (res.width > img.cols || res.height > img.rows) continue;
            
            cv::Mat test_img;
            cv::resize(img, test_img, res);
            
            auto result = benchmark_detector(detector_opt, test_img, 5, 30);
            auto dets = detector_opt.detect(test_img);
            
            print_table_row(res, result, dets.size());
        }
        
        std::cout << std::string(90, '=') << std::endl;
        
        // ========== TEST 4: Throughput Máximo ==========
        spdlog::info("\n⚡ TEST 4: Throughput Máximo");
        auto throughput = benchmark_detector(detector_opt, test_1080p, 20, 200);
        
        spdlog::info("  {} frames procesados", throughput.frames);
        spdlog::info("  Tiempo promedio: {:.2f}ms", throughput.avg_ms);
        spdlog::info("  Throughput: {:.1f} FPS", throughput.fps);
        spdlog::info("  Rango: {:.2f}ms - {:.2f}ms", throughput.min_ms, throughput.max_ms);
        
        // ========== Recomendaciones ==========
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    RECOMENDACIONES                         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        
        if (optimized.fps >= 30) {
            spdlog::info("✓ Performance excelente para video en tiempo real");
        } else if (optimized.fps >= 20) {
            spdlog::info("⚠️  Performance aceptable, considera:");
            spdlog::info("   - Reducir resolución de entrada");
            spdlog::info("   - Usar modelo SCRFD 500M (más ligero)");
        } else {
            spdlog::warn("❌ Performance insuficiente para tiempo real");
            spdlog::warn("   Acciones recomendadas:");
            spdlog::warn("   1. Verificar que engine sea FP16:");
            spdlog::warn("      trtexec --loadEngine={} --dumpProfile", engine_path);
            spdlog::warn("   2. Verificar OpenCV-CUDA:");
            spdlog::warn("      python3 -c 'import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())'");
            spdlog::warn("   3. Monitorear GPU:");
            spdlog::warn("      nvidia-smi dmon -i 0");
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
    
    return 0;
}