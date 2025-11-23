// ============= test/test_recognizer.cpp =============
#include "recognition/recognizer.hpp"  // ⭐ CAMBIO
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <vector>
#include <numeric>
#include <filesystem>

// -----------------------------------------------------
// Helpers
// -----------------------------------------------------
template<typename T>
double mean(const std::vector<T>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// -----------------------------------------------------
// Benchmark
// -----------------------------------------------------
void benchmark(const std::string& engine_path,
               const std::vector<cv::Mat>& faces,
               int warmup = 10,
               int iterations = 100)
{
    spdlog::info("🎭 Benchmark ArcFace");
    spdlog::info("   Engine : {}", engine_path);
    spdlog::info("   Faces  : {}", faces.size());
    spdlog::info("   GPU pre: true");

    FaceRecognizer recognizer(engine_path, true);

    // warmup
    for (int i = 0; i < warmup; ++i)
        recognizer.extract_embedding(faces[i % faces.size()]);

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i)
    {
        const auto& face = faces[i % faces.size()];
        auto t0 = std::chrono::high_resolution_clock::now();
        auto emb = recognizer.extract_embedding(face);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double avg = mean(times);
    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());

    spdlog::info("------------------------------------------------");
    spdlog::info("Average : {:.2f} ms", avg);
    spdlog::info("Min     : {:.2f} ms", min);
    spdlog::info("Max     : {:.2f} ms", max);
    spdlog::info("Throughput: {:.1f} FPS", 1000.0 / avg);
}

// -----------------------------------------------------
// Sanity check (same person → high similarity)
// -----------------------------------------------------
void sanity_check(FaceRecognizer& rec, const cv::Mat& face)
{
    auto e1 = rec.extract_embedding(face);
    auto e2 = rec.extract_embedding(face);
    float sim = FaceRecognizer::compare(e1, e2);
    spdlog::info("Self-similarity: {:.4f} (should be ~1.0)", sim);
}

// -----------------------------------------------------
// main
// -----------------------------------------------------
int main(int argc, char* argv[])
{
    spdlog::set_pattern("[%H:%M:%S] %v");
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <recognizer.engine> <image|(folder)> [--bench-iters N]\n";
        return 1;
    }

    std::string engine = argv[1];
    std::string path   = argv[2];
    int bench_iters = 100;
    if (argc >= 5 && std::string(argv[3]) == "--bench-iters")
        bench_iters = std::stoi(argv[4]);

    // cargar imagen o carpeta
    std::vector<cv::Mat> faces;
    if (std::filesystem::is_directory(path))
    {
        for (const auto& entry : std::filesystem::directory_iterator(path))
        {
            cv::Mat img = cv::imread(entry.path());
            if (!img.empty()) faces.push_back(img);
        }
    }
    else
    {
        cv::Mat img = cv::imread(path);
        if (!img.empty()) faces.push_back(img);
    }

    if (faces.empty())
    {
        spdlog::error("No valid images found");
        return 1;
    }

    try
    {
        FaceRecognizer rec(engine, true);
        sanity_check(rec, faces[0]);
        benchmark(engine, faces, 10, bench_iters);
    }
    catch (const std::exception& e)
    {
        spdlog::error("{}", e.what());
        return 1;
    }
    return 0;
}