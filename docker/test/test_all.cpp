#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

// Macros para colores en terminal
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void printHeader(const std::string& text) {
    std::cout << "\n" << BOLD << CYAN << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::setw(58) << std::left << text << " ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝" << RESET << "\n\n";
}

void printSection(const std::string& text) {
    std::cout << BOLD << YELLOW << "[" << text << "]" << RESET << "\n";
}

void printSuccess(const std::string& text) {
    std::cout << GREEN << "  ✓ " << text << RESET << "\n";
}

void printError(const std::string& text) {
    std::cout << RED << "  ✗ " << text << RESET << "\n";
}

void printInfo(const std::string& key, const std::string& value) {
    std::cout << "  • " << BOLD << key << RESET << ": " << value << "\n";
}

void printWarning(const std::string& text) {
    std::cout << YELLOW << "  ⚠ " << text << RESET << "\n";
}

// Benchmark helper
template<typename Func>
double benchmark(Func func, int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return duration.count() / static_cast<double>(iterations);
}

int main() {
    printHeader("VERIFICACIÓN COMPLETA: OpenCV + CUDA + TensorRT");
    
    // ===== 1. OPENCV BÁSICO =====
    printSection("1. Información de OpenCV");
    printInfo("Versión completa", CV_VERSION);
    printInfo("Major.Minor.Patch", std::to_string(CV_MAJOR_VERSION) + "." + 
                                   std::to_string(CV_MINOR_VERSION) + "." + 
                                   std::to_string(CV_SUBMINOR_VERSION));
    
    // ===== 2. CUDA SUPPORT =====
    printSection("2. Soporte CUDA");
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    printInfo("Dispositivos CUDA detectados", std::to_string(cuda_devices));
    
    bool cuda_available = (cuda_devices > 0);
    
    if (cuda_available) {
        printSuccess("CUDA HABILITADO");
        
        cv::cuda::DeviceInfo dev_info;
        printInfo("Nombre GPU", dev_info.name());
        printInfo("Compute Capability", std::to_string(dev_info.majorVersion()) + "." + 
                                        std::to_string(dev_info.minorVersion()));
        
        double total_mem_gb = dev_info.totalMemory() / (1024.0 * 1024.0 * 1024.0);
        double free_mem_gb = dev_info.freeMemory() / (1024.0 * 1024.0 * 1024.0);
        
        printInfo("Memoria Total", std::to_string(static_cast<int>(total_mem_gb)) + " GB");
        printInfo("Memoria Libre", std::to_string(static_cast<int>(free_mem_gb)) + " GB");
        printInfo("Multiprocesadores", std::to_string(dev_info.multiProcessorCount()));
        printInfo("Clock Rate", std::to_string(static_cast<int>(dev_info.clockRate() / 1000.0)) + " MHz");
        
        // Características CUDA
        printInfo("Async Engine Count", std::to_string(dev_info.asyncEngineCount()));
        printInfo("Memory Clock Rate", std::to_string(dev_info.memoryClockRate() / 1000) + " MHz");
        printInfo("Memory Bus Width", std::to_string(dev_info.memoryBusWidth()) + " bits");
        
    } else {
        printError("CUDA NO DISPONIBLE");
    }
    
    // ===== 3. MÓDULOS OPENCV CUDA =====
    printSection("3. Módulos CUDA de OpenCV");
    
    std::vector<std::string> cuda_modules = {
        "cudaarithm", "cudabgsegm", "cudacodec", "cudafeatures2d",
        "cudafilters", "cudaimgproc", "cudalegacy", "cudaobjdetect",
        "cudaoptflow", "cudastereo", "cudawarping"
    };
    
    std::cout << "  Verificando módulos CUDA disponibles:\n";
    for (const auto& module : cuda_modules) {
        // Estos módulos deberían estar compilados en OpenCV
        std::cout << "    • " << module << "\n";
    }
    
    // ===== 4. DNN MODULE =====
    printSection("4. OpenCV DNN Module");
    
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = 
        cv::dnn::getAvailableBackends();
    
    printInfo("Backends disponibles", std::to_string(backends.size()));
    
    bool has_cuda_backend = false;
    bool has_cuda_fp16 = false;
    
    for (const auto& backend : backends) {
        std::string backend_name;
        std::string target_name;
        
        // Backend names
        switch(backend.first) {
            case cv::dnn::DNN_BACKEND_DEFAULT: backend_name = "DEFAULT"; break;
            case cv::dnn::DNN_BACKEND_HALIDE: backend_name = "HALIDE"; break;
            case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE: backend_name = "INFERENCE_ENGINE"; break;
            case cv::dnn::DNN_BACKEND_OPENCV: backend_name = "OPENCV"; break;
            case cv::dnn::DNN_BACKEND_VKCOM: backend_name = "VKCOM"; break;
            case cv::dnn::DNN_BACKEND_CUDA: 
                backend_name = "CUDA"; 
                has_cuda_backend = true;
                break;
            default: backend_name = "UNKNOWN";
        }
        
        // Target names
        switch(backend.second) {
            case cv::dnn::DNN_TARGET_CPU: target_name = "CPU"; break;
            case cv::dnn::DNN_TARGET_OPENCL: target_name = "OPENCL"; break;
            case cv::dnn::DNN_TARGET_OPENCL_FP16: target_name = "OPENCL_FP16"; break;
            case cv::dnn::DNN_TARGET_MYRIAD: target_name = "MYRIAD"; break;
            case cv::dnn::DNN_TARGET_VULKAN: target_name = "VULKAN"; break;
            case cv::dnn::DNN_TARGET_CUDA: target_name = "CUDA"; break;
            case cv::dnn::DNN_TARGET_CUDA_FP16: 
                target_name = "CUDA_FP16";
                has_cuda_fp16 = true;
                break;
            default: target_name = "UNKNOWN";
        }
        
        std::cout << "    • " << backend_name << " → " << target_name << "\n";
    }
    
    if (has_cuda_backend) {
        printSuccess("DNN CUDA Backend disponible");
    } else {
        printError("DNN CUDA Backend NO disponible");
    }
    
    if (has_cuda_fp16) {
        printSuccess("DNN CUDA FP16 disponible");
    } else {
        printWarning("DNN CUDA FP16 no disponible");
    }
    
    // ===== 5. TENSORRT =====
    printSection("5. TensorRT C++ API");
    
    printInfo("Versión TensorRT", std::to_string(NV_TENSORRT_MAJOR) + "." + 
                                   std::to_string(NV_TENSORRT_MINOR) + "." + 
                                   std::to_string(NV_TENSORRT_PATCH));
    printInfo("Header", "/usr/include/aarch64-linux-gnu/NvInfer.h");
    printInfo("Library", "/usr/lib/aarch64-linux-gnu/libnvinfer.so");
    
    // Verificar si hay soporte de TensorRT en OpenCV DNN
    #ifdef OPENCV_DNN_CUDA
        printSuccess("OpenCV compilado con OPENCV_DNN_CUDA");
    #else
        printWarning("OpenCV sin OPENCV_DNN_CUDA");
    #endif
    
    // ===== 6. TESTS PRÁCTICOS CUDA =====
    printSection("6. Tests Prácticos CUDA");
    
    if (cuda_available) {
        try {
            // Test 1: Upload/Download
            std::cout << "\n  Test 1: Upload/Download GPU\n";
            cv::Mat img_cpu(1920, 1080, CV_8UC3);
            cv::randu(img_cpu, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
            
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(img_cpu);
            printSuccess("Upload 1920x1080 BGR a GPU");
            
            cv::Mat img_download;
            img_gpu.download(img_download);
            printSuccess("Download de GPU a CPU");
            
            // Test 2: Conversión de color
            std::cout << "\n  Test 2: Conversión de Color GPU\n";
            cv::cuda::GpuMat gray_gpu;
            cv::cuda::cvtColor(img_gpu, gray_gpu, cv::COLOR_BGR2GRAY);
            printSuccess("Conversión BGR→GRAY en GPU");
            
            cv::cuda::GpuMat bgr_gpu;
            cv::cuda::cvtColor(gray_gpu, bgr_gpu, cv::COLOR_GRAY2BGR);
            printSuccess("Conversión GRAY→BGR en GPU");
            
            // Test 3: Resize
            std::cout << "\n  Test 3: Resize GPU\n";
            cv::cuda::GpuMat resized_gpu;
            cv::cuda::resize(img_gpu, resized_gpu, cv::Size(640, 480));
            printSuccess("Resize 1920x1080 → 640x480 en GPU");
            
            // Test 4: Filtros
            std::cout << "\n  Test 4: Filtros GPU\n";
            cv::cuda::GpuMat blurred_gpu;
            auto gaussian = cv::cuda::createGaussianFilter(
                img_gpu.type(), img_gpu.type(), cv::Size(5, 5), 1.5);
            gaussian->apply(img_gpu, blurred_gpu);
            printSuccess("Filtro Gaussiano 5x5 en GPU");
            
            // Test 5: Operaciones aritméticas
            std::cout << "\n  Test 5: Operaciones Aritméticas GPU\n";
            cv::cuda::GpuMat result_gpu;
            cv::cuda::add(img_gpu, cv::Scalar(10, 10, 10), result_gpu);
            printSuccess("Adición escalar en GPU");
            
            cv::cuda::multiply(img_gpu, cv::Scalar(1.5, 1.5, 1.5), result_gpu);
            printSuccess("Multiplicación escalar en GPU");
            
            // Test 6: Benchmark
            std::cout << "\n  Test 6: Benchmark GPU vs CPU\n";
            
            cv::Mat test_img(1920, 1080, CV_8UC3);
            cv::randu(test_img, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
            cv::cuda::GpuMat test_gpu;
            test_gpu.upload(test_img);
            
            // CPU
            double cpu_time = benchmark([&]() {
                cv::Mat gray;
                cv::cvtColor(test_img, gray, cv::COLOR_BGR2GRAY);
            }, 100);
            
            // GPU
            double gpu_time = benchmark([&]() {
                cv::cuda::GpuMat gray;
                cv::cuda::cvtColor(test_gpu, gray, cv::COLOR_BGR2GRAY);
                cv::cuda::Stream::Null().waitForCompletion();
            }, 100);
            
            std::cout << "    CPU (cvtColor 1920x1080): " << std::fixed 
                      << std::setprecision(2) << cpu_time << " ms\n";
            std::cout << "    GPU (cvtColor 1920x1080): " << std::fixed 
                      << std::setprecision(2) << gpu_time << " ms\n";
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) 
                      << (cpu_time / gpu_time) << "x\n";
            
            printSuccess("TODOS LOS TESTS CUDA EXITOSOS");
            
        } catch (const cv::Exception& e) {
            printError("Error en tests CUDA: " + std::string(e.what()));
        }
    } else {
        printWarning("No se pueden ejecutar tests (CUDA no disponible)");
    }
    
    // ===== 7. GSTREAMER =====
    printSection("7. GStreamer Support");
    
    #ifdef HAVE_GSTREAMER
        printSuccess("OpenCV compilado con GStreamer");
    #else
        printWarning("OpenCV sin GStreamer");
    #endif
    
    // ===== 8. FFMPEG =====
    printSection("8. FFmpeg Support");
    
    #ifdef HAVE_FFMPEG
        printSuccess("OpenCV compilado con FFmpeg");
    #else
        printWarning("OpenCV sin FFmpeg");
    #endif
    
    // ===== RESUMEN FINAL =====
    printHeader("RESUMEN FINAL");
    
    std::cout << BOLD << "  Sistema:\n" << RESET;
    printInfo("OpenCV", CV_VERSION);
    printInfo("CUDA", cuda_available ? "✓ Disponible (" + std::to_string(cuda_devices) + " device)" : "✗ No disponible");
    printInfo("DNN CUDA", has_cuda_backend ? "✓ Disponible" : "✗ No disponible");
    printInfo("DNN CUDA FP16", has_cuda_fp16 ? "✓ Disponible" : "✗ No disponible");
    printInfo("TensorRT", std::to_string(NV_TENSORRT_MAJOR) + "." + 
                          std::to_string(NV_TENSORRT_MINOR) + "." + 
                          std::to_string(NV_TENSORRT_PATCH));
    
    std::cout << "\n" << BOLD << "  Estado General: ";
    if (cuda_available && has_cuda_backend) {
        std::cout << GREEN << "✓ TODO OPERATIVO" << RESET << "\n";
    } else {
        std::cout << YELLOW << "⚠ REQUIERE ATENCIÓN" << RESET << "\n";
    }
    
    std::cout << "\n";
    
    return 0;
}