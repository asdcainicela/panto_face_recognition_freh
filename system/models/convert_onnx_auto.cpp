#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <Python.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            spdlog::warn("[TensorRT] {}", msg);
        }
    }
};

struct OnnxMetadata {
    std::string inputName;
    std::vector<int64_t> inputShape;
    std::string outputName;
    std::vector<int64_t> outputShape;
    bool valid = false;
    std::string error;
};

class PythonOnnxInspector {
private:
    PyObject* pModule = nullptr;
    PyObject* pOnnxModule = nullptr;
    
public:
    PythonOnnxInspector() {
        Py_Initialize();
        
        // Import sys para agregar path si es necesario
        PyRun_SimpleString("import sys");
        
        // Import onnx
        pOnnxModule = PyImport_ImportModule("onnx");
        if (!pOnnxModule) {
            spdlog::error("No se pudo importar módulo 'onnx'");
            PyErr_Print();
        }
    }
    
    ~PythonOnnxInspector() {
        Py_XDECREF(pOnnxModule);
        Py_XDECREF(pModule);
        Py_Finalize();
    }
    
    OnnxMetadata inspect(const std::string& onnxPath) {
        OnnxMetadata meta;
        
        if (!pOnnxModule) {
            meta.error = "Módulo ONNX no disponible";
            return meta;
        }
        
        // Construir código Python para inspección
        std::string pythonCode = R"(
import onnx

def inspect_onnx(path):
    try:
        model = onnx.load(path)
        
        # Input info
        input_tensor = model.graph.input[0]
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        
        # Output info
        output_tensor = model.graph.output[0]
        output_name = output_tensor.name
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        
        return {
            'input_name': input_name,
            'input_shape': input_shape,
            'output_name': output_name,
            'output_shape': output_shape,
            'valid': True
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

result = inspect_onnx(')" + onnxPath + R"(')
)";
        
        // Ejecutar código Python
        PyObject* pMain = PyImport_AddModule("__main__");
        PyObject* pDict = PyModule_GetDict(pMain);
        
        PyRun_String(pythonCode.c_str(), Py_file_input, pDict, pDict);
        
        // Obtener resultado
        PyObject* pResult = PyDict_GetItemString(pDict, "result");
        
        if (!pResult || !PyDict_Check(pResult)) {
            meta.error = "Error ejecutando inspección Python";
            PyErr_Print();
            return meta;
        }
        
        // Extraer 'valid'
        PyObject* pValid = PyDict_GetItemString(pResult, "valid");
        if (pValid && PyBool_Check(pValid)) {
            meta.valid = (pValid == Py_True);
        }
        
        if (!meta.valid) {
            // Extraer error
            PyObject* pError = PyDict_GetItemString(pResult, "error");
            if (pError && PyUnicode_Check(pError)) {
                meta.error = PyUnicode_AsUTF8(pError);
            }
            return meta;
        }
        
        // Extraer input_name
        PyObject* pInputName = PyDict_GetItemString(pResult, "input_name");
        if (pInputName && PyUnicode_Check(pInputName)) {
            meta.inputName = PyUnicode_AsUTF8(pInputName);
        }
        
        // Extraer input_shape
        PyObject* pInputShape = PyDict_GetItemString(pResult, "input_shape");
        if (pInputShape && PyList_Check(pInputShape)) {
            Py_ssize_t size = PyList_Size(pInputShape);
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject* item = PyList_GetItem(pInputShape, i);
                if (PyLong_Check(item)) {
                    meta.inputShape.push_back(PyLong_AsLongLong(item));
                }
            }
        }
        
        // Extraer output_name
        PyObject* pOutputName = PyDict_GetItemString(pResult, "output_name");
        if (pOutputName && PyUnicode_Check(pOutputName)) {
            meta.outputName = PyUnicode_AsUTF8(pOutputName);
        }
        
        // Extraer output_shape
        PyObject* pOutputShape = PyDict_GetItemString(pResult, "output_shape");
        if (pOutputShape && PyList_Check(pOutputShape)) {
            Py_ssize_t size = PyList_Size(pOutputShape);
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject* item = PyList_GetItem(pOutputShape, i);
                if (PyLong_Check(item)) {
                    meta.outputShape.push_back(PyLong_AsLongLong(item));
                }
            }
        }
        
        return meta;
    }
};

std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("No se pudo abrir: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::string buffer(size, '\0');
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Error leyendo: " + filename);
    }
    return buffer;
}

void printShape(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        spdlog::error("Uso: {} <input.onnx> <output.engine>", argv[0]);
        return 1;
    }
    
    std::string onnxFile = argv[1];
    std::string engineFile = argv[2];
    
    spdlog::info("=== TensorRT Auto Converter ===");
    spdlog::info("ONNX Input:  {}", onnxFile);
    spdlog::info("Engine Output: {}", engineFile);
    
    // Verificar que existe el archivo
    std::ifstream testFile(onnxFile);
    if (!testFile.good()) {
        spdlog::error("Archivo ONNX no encontrado: {}", onnxFile);
        return 1;
    }
    testFile.close();
    
    // Inspeccionar ONNX con Python
    spdlog::info("Inspeccionando modelo ONNX...");
    PythonOnnxInspector inspector;
    OnnxMetadata meta = inspector.inspect(onnxFile);
    
    if (!meta.valid) {
        spdlog::error("Error inspeccionando ONNX: {}", meta.error);
        return 1;
    }
    
    // Mostrar información detectada
    spdlog::info("✓ Metadata detectada:");
    std::cout << "  Input:  " << meta.inputName << " ";
    printShape(meta.inputShape);
    std::cout << std::endl;
    
    std::cout << "  Output: " << meta.outputName << " ";
    printShape(meta.outputShape);
    std::cout << std::endl;
    
    // Validar shape de input
    if (meta.inputShape.size() != 4) {
        spdlog::error("Se esperaba input shape de 4 dimensiones (NCHW)");
        return 1;
    }
    
    // Crear TensorRT builder
    TRTLogger logger;
    
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    if (!builder) {
        spdlog::error("No se pudo crear IBuilder");
        return 1;
    }
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    
    if (!network) {
        spdlog::error("No se pudo crear INetworkDefinition");
        return 1;
    }
    
    // Parser ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));
    
    if (!parser) {
        spdlog::error("No se pudo crear ONNX parser");
        return 1;
    }
    
    spdlog::info("Parseando modelo ONNX...");
    std::string onnxModel = readFile(onnxFile);
    
    if (!parser->parse(onnxModel.data(), onnxModel.size())) {
        spdlog::error("Error parseando ONNX");
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            spdlog::error("  - {}", parser->getError(i)->desc());
        }
        return 1;
    }
    
    // Configurar builder
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        spdlog::error("No se pudo crear IBuilderConfig");
        return 1;
    }
    
    // Memoria de trabajo: 3GB
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30);
    
    // Habilitar FP16 si está disponible
    if (builder->platformHasFastFp16()) {
        spdlog::info("✓ FP16 habilitado");
        config->setFlag(BuilderFlag::kFP16);
    } else {
        spdlog::warn("FP16 no disponible en esta plataforma");
    }
    
    // Aplicar dimensiones detectadas
    auto inputTensor = network->getInput(0);
    inputTensor->setName(meta.inputName.c_str());
    
    Dims inputDims;
    inputDims.nbDims = meta.inputShape.size();
    for (size_t i = 0; i < meta.inputShape.size(); ++i) {
        inputDims.d[i] = meta.inputShape[i];
    }
    inputTensor->setDimensions(inputDims);
    
    // Construir engine
    spdlog::info("Construyendo TensorRT engine (esto puede tardar varios minutos)...");
    auto engine = std::unique_ptr<ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config));
    
    if (!engine) {
        spdlog::error("Error construyendo engine TensorRT");
        return 1;
    }
    
    // Serializar y guardar
    spdlog::info("Serializando engine...");
    auto serialized = std::unique_ptr<IHostMemory>(engine->serialize());
    
    if (!serialized) {
        spdlog::error("Error serializando engine");
        return 1;
    }
    
    std::ofstream outFile(engineFile, std::ios::binary);
    if (!outFile) {
        spdlog::error("No se pudo crear archivo: {}", engineFile);
        return 1;
    }
    
    outFile.write(reinterpret_cast<const char*>(serialized->data()), 
                  serialized->size());
    outFile.close();
    
    spdlog::info("✓ Engine guardado exitosamente: {}", engineFile);
    spdlog::info("Tamaño: {} MB", serialized->size() / (1024.0 * 1024.0));
    
    return 0;
}