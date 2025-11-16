//./convert_onnx_auto arcface_r100.onnx engines/arcface_r100.onx

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
#include <cstdio>

using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            spdlog::warn("[TensorRT] {}", msg);
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

/* ----------  Python helper  ---------- */
class PythonOnnxInspector {
public:
    PythonOnnxInspector() {
        Py_Initialize();
        PyRun_SimpleString("import sys");
        pOnnx = PyImport_ImportModule("onnx");
        if (!pOnnx) { PyErr_Print(); spdlog::error("onnx no disponible"); }
    }
    ~PythonOnnxInspector() {
        Py_XDECREF(pOnnx);
        Py_Finalize();
    }
    OnnxMetadata inspect(const std::string& path) {
        OnnxMetadata m;
        if (!pOnnx) { m.error = "onnx no disponible"; return m; }

        char code[4096];
        std::snprintf(code, sizeof(code), R"(
import onnx
def inspect(path):
    try:
        g = onnx.load(path).graph
        inp = g.input[0]
        out = g.output[0]
        return {
            'valid':True,
            'input_name': inp.name,
            'input_shape': [d.dim_value for d in inp.type.tensor_type.shape.dim],
            'output_name': out.name,
            'output_shape': [d.dim_value for d in out.type.tensor_type.shape.dim]
        }
    except Exception as e:
        return {'valid':False,'error':str(e)}
result = inspect(r'%s')
)", path.c_str());

        PyObject* main = PyImport_AddModule("__main__");
        PyObject* dict = PyModule_GetDict(main);
        PyRun_String(code, Py_file_input, dict, dict);
        PyObject* res = PyDict_GetItemString(dict, "result");
        if (!res || !PyDict_Check(res)) { m.error = "python error"; return m; }

        PyObject* pValid = PyDict_GetItemString(res, "valid");
        m.valid = pValid && pValid == Py_True;
        if (!m.valid) {
            PyObject* pErr = PyDict_GetItemString(res, "error");
            if (pErr) m.error = PyUnicode_AsUTF8(pErr);
            return m;
        }
        auto getStr = [&](const char* k){
            PyObject* o = PyDict_GetItemString(res, k);
            return std::string(o ? PyUnicode_AsUTF8(o) : "");
        };
        auto getVec = [&](const char* k){
            std::vector<int64_t> v;
            PyObject* o = PyDict_GetItemString(res, k);
            if (PyList_Check(o))
                for (Py_ssize_t i = 0; i < PyList_Size(o); ++i)
                    v.push_back(PyLong_AsLong(PyList_GetItem(o, i)));
            return v;
        };
        m.inputName  = getStr("input_name");
        m.inputShape = getVec("input_shape");
        m.outputName = getStr("output_name");
        m.outputShape= getVec("output_shape");
        return m;
    }
private:
    PyObject* pOnnx = nullptr;
};

/* ----------  helpers  ---------- */
std::string readFile(const std::string& file) {
    std::ifstream f(file, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("no se puede abrir " + file);
    auto sz = f.tellg();
    f.seekg(0);
    std::string buf(sz, '\0');
    if (!f.read(buf.data(), sz)) throw std::runtime_error("lectura falló");
    return buf;
}
void printShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i)
        std::cout << s[i] << (i + 1 < s.size() ? ", " : "");
    std::cout << "]";
}

/* ----------  main  ---------- */
int main(int argc, char** argv) {
    if (argc != 3) {
        spdlog::error("uso: {} <model.onnx> <model.engine>", argv[0]);
        return 1;
    }
    std::string onnxPath = argv[1];
    std::string engPath  = argv[2];

    spdlog::info("=== TensorRT auto-converter ===");
    spdlog::info("ONNX : {}", onnxPath);
    spdlog::info("ENGINE: {}", engPath);

    TRTLogger logger;
    PythonOnnxInspector py;

    auto meta = py.inspect(onnxPath);
    if (!meta.valid) {
        spdlog::error("inspección onnx: {}", meta.error);
        return 1;
    }
    spdlog::info("metadatos:");
    std::cout << "  input  : " << meta.inputName << " "; printShape(meta.inputShape); std::cout << "\n";
    std::cout << "  output : " << meta.outputName << " "; printShape(meta.outputShape); std::cout << "\n";

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    if (!builder) { spdlog::error("sin builder"); return 1; }

    const auto flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flag));
    if (!network) { spdlog::error("sin network"); return 1; }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser) { spdlog::error("sin parser"); return 1; }

    std::string onnxBlob = readFile(onnxPath);
    if (!parser->parse(onnxBlob.data(), onnxBlob.size())) {
        spdlog::error("error parseando onnx");
        for (int i = 0; i < parser->getNbErrors(); ++i)
            spdlog::error("  - {}", parser->getError(i)->desc());
        return 1;
    }

    /* ----------  ¿dimensiones dinámicas?  ---------- */
    auto* inputT = network->getInput(0);
    Dims orig = inputT->getDimensions();
    bool dynamic = false;
    for (int i = 0; i < orig.nbDims; ++i)
        if (orig.d[i] <= 0) { dynamic = true; break; }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 3ULL << 30);
    if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
        spdlog::info("FP16 habilitado");
    }

    if (dynamic) {
        spdlog::warn("dimensiones dinámicas detectadas – configurando perfil");
        auto ask = [&](const char* name, int64_t& val) {
            if (val <= 0) {
                std::cout << "  " << name << " : ";
                std::cin  >> val;
            }
        };
        ask("Batch (N)", meta.inputShape[0]);
        if (meta.inputShape.size() >= 2) ask("Channels (C)", meta.inputShape[1]);
        if (meta.inputShape.size() >= 3) ask("Height  (H)", meta.inputShape[2]);
        if (meta.inputShape.size() >= 4) ask("Width   (W)", meta.inputShape[3]);

        Dims tgt;
        tgt.nbDims = meta.inputShape.size();
        for (size_t i = 0; i < meta.inputShape.size(); ++i) tgt.d[i] = meta.inputShape[i];

        auto* profile = builder->createOptimizationProfile();
        profile->setDimensions(inputT->getName(), OptProfileSelector::kMIN, tgt);
        profile->setDimensions(inputT->getName(), OptProfileSelector::kOPT, tgt);
        profile->setDimensions(inputT->getName(), OptProfileSelector::kMAX, tgt);
        config->addOptimizationProfile(profile);
        spdlog::info("perfil fijado a [{} {} {} {}]",
                     tgt.d[0], tgt.d[1], tgt.d[2], tgt.d[3]);
    }

    spdlog::info("construyendo engine (puede tardar varios minutos)...");
    auto serialized = std::unique_ptr<IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
    if (!serialized) { spdlog::error("buildSerializedNetwork falló"); return 1; }

    std::ofstream out(engPath, std::ios::binary);
    if (!out) { spdlog::error("no se puede crear {}", engPath); return 1; }
    out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    spdlog::info("✓ engine guardado: {}  ({} MB)", engPath, serialized->size() / (1024.0 * 1024.0));
    return 0;
}