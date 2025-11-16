/*
 ["retinaface.onnx"]="retinaface_fp16.engine"
 ["arcface_r100.onnx"]="arcface_r100_fp16.engine"
*/

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <spdlog/spdlog.h>
#include <Python.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TRT] {}", msg);
        }
    }
};

struct OnnxMetadata {
    std::string inputName;
    std::vector<int> inputShape;
    bool valid = false;
};

// Inspecciona ONNX con Python embedding
OnnxMetadata inspectOnnx(const std::string& onnxFile) {
    OnnxMetadata meta;
    
    Py_Initialize();
    
    std::string pythonCode = R"(
import onnx
import json

def inspect(path):
    model = onnx.load(path)
    inp = model.graph.input[0]
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    return {"name": inp.name, "shape": shape}

result = inspect(')" + onnxFile + R"(')
print(json.dumps(result))
)";
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.stdout = open('/tmp/onnx_meta.txt', 'w')");
    PyRun_SimpleString(pythonCode.c_str());
    
    Py_Finalize();
    
    // Lee resultado
    std::ifstream metaFile("/tmp/onnx_meta.txt");
    if (metaFile.good()) {
        std::string line;
        std::getline(metaFile, line);
        
        // Parse JSON básico (asume formato {"name":"X","shape":[1,3,H,W]})
        size_t namePos = line.find("\"name\":\"") + 8;
        size_t nameEnd = line.find("\"", namePos);
        meta.inputName = line.substr(namePos, nameEnd - namePos);
        
        // Extrae shape (simplificado)
        size_t shapePos = line.find("[", line.find("shape")) + 1;
        size_t shapeEnd = line.find("]", shapePos);
        std::string shapeStr = line.substr(shapePos, shapeEnd - shapePos);
        
        // Parse números
        std::stringstream ss(shapeStr);
        std::string num;
        while (std::getline(ss, num, ',')) {
            meta.inputShape.push_back(std::stoi(num));
        }
        
        meta.valid = true;
    }
    
    return meta;
}

std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("No se pudo abrir archivo: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Error leyendo archivo");
    return buffer;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        spdlog::error("Uso: {} <input.onnx> <output.engine>", argv[0]);
        return 1;
    }
    
    std::string onnxFile = argv[1];
    std::string engineFile = argv[2];
    
    spdlog::info("=== Auto TensorRT Converter ===");
    spdlog::info("Inspeccionando ONNX: {}", onnxFile);
    
    // Auto-detectar metadata con Python
    OnnxMetadata meta = inspectOnnx(onnxFile);
    
    if (!meta.valid) {
        spdlog::error("No se pudo inspeccionar el ONNX");
        return 1;
    }
    
    spdlog::info("Input detectado: {}", meta.inputName);
    spdlog::info("Shape: [{},{},{},{}]", 
        meta.inputShape[0], meta.inputShape[1], 
        meta.inputShape[2], meta.inputShape[3]);
    
    TRTLogger logger;
    
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
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30);
    
    if (builder->platformHasFastFp16()) {
        spdlog::info("FP16 habilitado");
        config->setFlag(BuilderFlag::kFP16);
    }
    
    // Aplicar shape detectado
    auto inputTensor = network->getInput(0);
    inputTensor->setName(meta.inputName.c_str());
    Dims inputDims{4, {meta.inputShape[0], meta.inputShape[1], 
                       meta.inputShape[2], meta.inputShape[3]}};
    inputTensor->setDimensions(inputDims);
    
    spdlog::info("Construyendo engine TensorRT...");
    auto engine = std::unique_ptr<ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config)
    );
    
    if (!engine) {
        spdlog::error("Error construyendo engine");
        return 1;
    }
    
    auto serialized = std::unique_ptr<IHostMemory>(engine->serialize());
    std::ofstream out(engineFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(serialized->data()), 
              serialized->size());
    
    spdlog::info("✓ Engine guardado: {}", engineFile);
    
    return 0;
}