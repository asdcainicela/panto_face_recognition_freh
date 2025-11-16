/*
 ["retinaface.onnx"]="retinaface_fp16.engine"
 ["arcface_r100.onnx"]="arcface_r100_fp16.engine"
*/

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace nvinfer1;

// Logger de TensorRT usando spdlog
class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TRT] {}", msg);
        }
    }
};

// Función para leer archivo binario
std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("No se pudo abrir archivo: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Error leyendo archivo");
    return buffer;
}

int main() {
    spdlog::info("=== TensorRT ONNX -> Engine Converter ===");

    TRTLogger logger;

    std::string onnxFile;
    spdlog::info("Ingrese el nombre del archivo ONNX:");
    std::cin >> onnxFile;

    std::ifstream f(onnxFile);
    if (!f.good()) {
        spdlog::error("Archivo ONNX no encontrado: {}", onnxFile);
        return 1;
    }

    std::string engineFile;
    spdlog::info("Ingrese el nombre de salida para el engine TensorRT:");
    std::cin >> engineFile;

    std::string inputName;
    spdlog::info("Ingrese el nombre del tensor de input (ej: input_0):");
    std::cin >> inputName;

    int C, H, W;
    spdlog::info("Ingrese las dimensiones del input (C H W):");
    std::cin >> C >> H >> W;

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    );

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger)
    );

    std::string onnxModel = readFile(onnxFile);

    if (!parser->parse(onnxModel.data(), onnxModel.size())) {
        spdlog::error("Error parseando ONNX");
        return 1;
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30); // 3GB
    if (builder->platformHasFastFp16())
        config->setFlag(BuilderFlag::kFP16);

    auto inputTensor = network->getInput(0);
    inputTensor->setName(inputName.c_str());
    Dims inputDims{4, {1, C, H, W}}; // batch=1
    inputTensor->setDimensions(inputDims);

    spdlog::info("Construyendo engine...");
    auto engine = std::unique_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        spdlog::error("Error construyendo engine");
        return 1;
    }

    auto serialized = std::unique_ptr<IHostMemory>(engine->serialize());
    std::ofstream out(engineFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    spdlog::info("Engine guardado en: {}", engineFile);

    return 0;
}
