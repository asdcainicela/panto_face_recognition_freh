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

void header(const std::string& text) {
    std::cout << "\n=== " << text << " ===\n\n";
}

void section(const std::string& text) {
    std::cout << "[" << text << "]\n";
}

void info(const std::string& k, const std::string& v) {
    std::cout << "  " << k << ": " << v << "\n";
}

void ok(const std::string& t) {
    std::cout << "  OK " << t << "\n";
}

void err(const std::string& t) {
    std::cout << "  ERROR " << t << "\n";
}

void warn(const std::string& t) {
    std::cout << "  WARNING " << t << "\n";
}

template<typename F>
double bench(F f, int n = 10) {
    auto s = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) f();
    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
    return d.count() / double(n);
}

int main() {
    header("VERIFICACION OPENCV CUDA TENSORRT");

    section("1. OpenCV");
    info("Version", CV_VERSION);
    info("Version major.minor.patch",
         std::to_string(CV_MAJOR_VERSION) + "." +
         std::to_string(CV_MINOR_VERSION) + "." +
         std::to_string(CV_SUBMINOR_VERSION));

    section("2. CUDA");
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    info("CUDA devices", std::to_string(cuda_devices));

    bool cuda_ok = cuda_devices > 0;
    if (cuda_ok) {
        ok("CUDA disponible");
        cv::cuda::DeviceInfo di;

        info("GPU", di.name());
        info("Compute capability",
             std::to_string(di.majorVersion()) + "." +
             std::to_string(di.minorVersion()));
        info("Memoria total (GB)", std::to_string(int(di.totalMemory() / 1e9)));
        info("Memoria libre (GB)", std::to_string(int(di.freeMemory() / 1e9)));
        info("Multiprocesadores", std::to_string(di.multiProcessorCount()));
        info("Clock MHz", std::to_string(int(di.clockRate() / 1000)));
        info("Async engines", std::to_string(di.asyncEngineCount()));
        info("Mem clock MHz", std::to_string(di.memoryClockRate() / 1000));
        info("Bus width bits", std::to_string(di.memoryBusWidth()));
    } else {
        err("CUDA no disponible");
    }

    section("3. Modulos CUDA");
    std::vector<std::string> cuda_modules = {
        "cudaarithm","cudabgsegm","cudacodec","cudafeatures2d",
        "cudafilters","cudaimgproc","cudalegacy","cudaobjdetect",
        "cudaoptflow","cudastereo","cudawarping"
    };
    for (auto& m : cuda_modules) std::cout << "  " << m << "\n";

    section("4. DNN");
    auto backends = cv::dnn::getAvailableBackends();
    info("Backends detectados", std::to_string(backends.size()));

    bool has_cuda_backend = false;
    bool has_cuda_fp16 = false;

    for (auto& b : backends) {
        std::string bn, tn;

        switch (b.first) {
            case cv::dnn::DNN_BACKEND_DEFAULT: bn = "DEFAULT"; break;
            case cv::dnn::DNN_BACKEND_HALIDE: bn = "HALIDE"; break;
            case cv::dnn::DNN_BACKEND_INFERENCE_ENGINE: bn = "IE"; break;
            case cv::dnn::DNN_BACKEND_OPENCV: bn = "OPENCV"; break;
            case cv::dnn::DNN_BACKEND_VKCOM: bn = "VKCOM"; break;
            case cv::dnn::DNN_BACKEND_CUDA: bn = "CUDA"; has_cuda_backend = true; break;
            default: bn = "UNKNOWN";
        }

        switch (b.second) {
            case cv::dnn::DNN_TARGET_CPU: tn = "CPU"; break;
            case cv::dnn::DNN_TARGET_OPENCL: tn = "OPENCL"; break;
            case cv::dnn::DNN_TARGET_OPENCL_FP16: tn = "OPENCL_FP16"; break;
            case cv::dnn::DNN_TARGET_VULKAN: tn = "VULKAN"; break;
            case cv::dnn::DNN_TARGET_CUDA: tn = "CUDA"; break;
            case cv::dnn::DNN_TARGET_CUDA_FP16: tn = "CUDA_FP16"; has_cuda_fp16 = true; break;
            default: tn = "UNKNOWN";
        }

        std::cout << "  " << bn << " -> " << tn << "\n";
    }

    if (has_cuda_backend) ok("DNN backend CUDA disponible");
    else warn("DNN backend CUDA no disponible");

    if (has_cuda_fp16) ok("DNN CUDA FP16 disponible");
    else warn("DNN CUDA FP16 no disponible");

    section("5. TensorRT");
    info("TensorRT version",
         std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH));
    info("NvInfer header", "/usr/include/aarch64-linux-gnu/NvInfer.h");
    info("NvInfer lib", "/usr/lib/aarch64-linux-gnu/libnvinfer.so");

    section("6. Tests CUDA");

    if (cuda_ok) {
        try {
            cv::Mat img(1920,1080,CV_8UC3);
            cv::randu(img, cv::Scalar(0,0,0), cv::Scalar(255,255,255));

            cv::cuda::GpuMat g;
            g.upload(img);
            ok("Upload");

            cv::Mat d;
            g.download(d);
            ok("Download");

            cv::cuda::GpuMat gray;
            cv::cuda::cvtColor(g, gray, cv::COLOR_BGR2GRAY);
            ok("cvtColor BGR->GRAY");

            cv::cuda::GpuMat r;
            cv::cuda::resize(g, r, cv::Size(640,480));
            ok("resize");

            auto gf = cv::cuda::createGaussianFilter(g.type(), g.type(), cv::Size(5,5), 1.5);
            cv::cuda::GpuMat gb;
            gf->apply(g, gb);
            ok("gaussian filter");

            cv::cuda::GpuMat ar;
            cv::cuda::add(g, cv::Scalar(10,10,10), ar);
            ok("add scalar");

            cv::cuda::GpuMat mul;
            cv::cuda::multiply(g, cv::Scalar(1.5,1.5,1.5), mul);
            ok("multiply scalar");

            double cpu_t = bench([&](){
                cv::Mat gr;
                cv::cvtColor(img, gr, cv::COLOR_BGR2GRAY);
            }, 100);

            double gpu_t = bench([&](){
                cv::cuda::GpuMat gr;
                cv::cuda::cvtColor(g, gr, cv::COLOR_BGR2GRAY);
                cv::cuda::Stream::Null().waitForCompletion();
            }, 100);

            info("CPU cvtColor ms", std::to_string(cpu_t));
            info("GPU cvtColor ms", std::to_string(gpu_t));
            info("Speedup", std::to_string(cpu_t / gpu_t));

        } catch (const cv::Exception& e) {
            err(std::string("Excepcion CUDA: ") + e.what());
        }
    } else {
        warn("CUDA no disponible, tests omitidos");
    }

    section("7. GStreamer");
#ifdef HAVE_GSTREAMER
    ok("GStreamer habilitado");
#else
    warn("GStreamer no habilitado");
#endif

    section("8. FFmpeg");
#ifdef HAVE_FFMPEG
    ok("FFmpeg habilitado");
#else
    warn("FFmpeg no habilitado");
#endif

    header("RESUMEN");

    info("OpenCV", CV_VERSION);
    info("CUDA", cuda_ok ? "Disponible" : "No disponible");
    info("DNN CUDA", has_cuda_backend ? "Si" : "No");
    info("DNN CUDA FP16", has_cuda_fp16 ? "Si" : "No");
    info("TensorRT",
        std::to_string(NV_TENSORRT_MAJOR) + "." +
        std::to_string(NV_TENSORRT_MINOR) + "." +
        std::to_string(NV_TENSORRT_PATCH));

    return 0;
}
