/*
./verify_engine engines/buffalo_l.engine
*/

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            spdlog::warn("[TensorRT] {}", msg);
    }
};

static std::string dimsToStr(const Dims& d) {
    std::string s = "[";
    for (int i = 0; i < d.nbDims; ++i) {
        s += std::to_string(d.d[i]);
        if (i + 1 < d.nbDims) s += ", ";
    }
    s += "]";
    return s;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        spdlog::error("uso: {} <model.engine>", argv[0]);
        return 1;
    }

    std::string engPath = argv[1];

    // 1. validar existencia y tamaño
    std::ifstream f(engPath, std::ios::binary | std::ios::ate);
    if (!f) {
        spdlog::error("archivo no encontrado: {}", engPath);
        return 1;
    }
    size_t sz = f.tellg();
    f.close();

    if (sz < 100) {
        spdlog::error("archivo demasiado pequeño ({} B) – engine corrupto", sz);
        return 1;
    }

    spdlog::info("tamaño engine: {:.2f} MB", sz / (1024.0 * 1024.0));

    // 2. cargar con TensorRT
    TRTLogger logger;
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    if (!runtime) {
        spdlog::error("no se pudo crear IRuntime");
        return 1;
    }

    std::vector<char> blob(sz);
    std::ifstream in(engPath, std::ios::binary);
    in.read(blob.data(), sz);
    in.close();

    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(blob.data(), blob.size()));

    if (!engine) {
        spdlog::error("deserialización fallida – engine corrupto");
        return 1;
    }

    spdlog::info("engine cargado correctamente");

    // 3. mostrar bindings
    int nb = engine->getNbBindings();
    spdlog::info("bindings: {}", nb);

    for (int i = 0; i < nb; ++i) {
        bool isIn = engine->bindingIsInput(i);
        const char* name = engine->getBindingName(i);
        Dims dims = engine->getBindingDimensions(i);
        DataType dtype = engine->getBindingDataType(i);

        spdlog::info("  [{}] {} {} tipo={} dims={}",
                     i,
                     isIn ? "INPUT" : "OUTPUT",
                     name,
                     static_cast<int>(dtype),
                     dimsToStr(dims));
    }

    // 4. crear contexto
    auto ctx = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    if (!ctx) {
        spdlog::error("no se pudo crear contexto");
        return 1;
    }

    spdlog::info("contexto creado – engine listo para inferir");

    return 0;
}
