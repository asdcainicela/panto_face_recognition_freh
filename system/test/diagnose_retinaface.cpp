#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_path = argc >= 2 ? argv[1] : "models/retinaface.onnx";
    std::string image_path = argc >= 3 ? argv[2] : "";
    
    spdlog::set_level(spdlog::level::info);
    std::cout << "=== DIAGNÓSTICO RETINAFACE ===" << std::endl;
    std::cout << "Modelo: " << model_path << std::endl;
    
    try {
        // Cargar modelo
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Diagnostic");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        
        Ort::Session session(env, model_path.c_str(), options);
        
        // Input info
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();
        
        std::cout << "\n=== INPUTS ===" << std::endl;
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "Input " << i << ": " << name.get() << std::endl;
            std::cout << "  Shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Type: " << tensor_info.GetElementType() << std::endl;
        }
        
        std::cout << "\n=== OUTPUTS ===" << std::endl;
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            std::cout << "Output " << i << ": " << name.get() << std::endl;
            std::cout << "  Shape: [";
            for (size_t j = 0; j < shape.size(); j++) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Type: " << tensor_info.GetElementType() << std::endl;
        }
        
        // Test con imagen si existe
        if (!image_path.empty()) {
            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                std::cerr << "No se pudo leer: " << image_path << std::endl;
                return 1;
            }
            
            std::cout << "\n=== PRUEBA CON IMAGEN ===" << std::endl;
            std::cout << "Imagen: " << img.cols << "x" << img.rows << std::endl;
            
            // Preprocess
            int target_w = 640, target_h = 640;
            cv::Mat resized, normalized;
            cv::resize(img, resized, cv::Size(target_w, target_h));
            resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
            cv::cvtColor(normalized, normalized, cv::COLOR_BGR2RGB);
            
            // HWC -> CHW
            std::vector<cv::Mat> channels(3);
            cv::split(normalized, channels);
            
            std::vector<float> input_data(1 * 3 * target_h * target_w);
            size_t single_size = target_h * target_w;
            for (int c = 0; c < 3; c++) {
                std::memcpy(input_data.data() + c * single_size, 
                           channels[c].data, 
                           single_size * sizeof(float));
            }
            
            // Input tensor
            std::vector<int64_t> input_shape = {1, 3, target_h, target_w};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data.data(), input_data.size(),
                input_shape.data(), input_shape.size()
            );
            
            // Get input/output names
            std::vector<const char*> input_names, output_names;
            for (size_t i = 0; i < num_inputs; i++) {
                auto name = session.GetInputNameAllocated(i, allocator);
                input_names.push_back(strdup(name.get()));
            }
            for (size_t i = 0; i < num_outputs; i++) {
                auto name = session.GetOutputNameAllocated(i, allocator);
                output_names.push_back(strdup(name.get()));
            }
            
            // Inference
            std::cout << "Ejecutando inferencia..." << std::endl;
            auto outputs = session.Run(
                Ort::RunOptions{nullptr},
                input_names.data(), &input_tensor, num_inputs,
                output_names.data(), num_outputs
            );
            
            std::cout << "\n=== RESULTADOS INFERENCIA ===" << std::endl;
            for (size_t i = 0; i < outputs.size(); i++) {
                auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
                size_t total = 1;
                for (auto s : shape) total *= s;
                
                std::cout << "Output " << i << " (" << output_names[i] << "):" << std::endl;
                std::cout << "  Shape: [";
                for (size_t j = 0; j < shape.size(); j++) {
                    std::cout << shape[j];
                    if (j < shape.size() - 1) std::cout << ", ";
                }
                std::cout << "] (total: " << total << " elementos)" << std::endl;
                
                // Mostrar primeros valores
                auto* data = outputs[i].GetTensorData<float>();
                std::cout << "  Primeros 10 valores: ";
                for (int j = 0; j < std::min(10, (int)total); j++) {
                    std::cout << data[j] << " ";
                }
                std::cout << std::endl;
                
                // Estadísticas
                float min_val = data[0], max_val = data[0], sum = 0;
                for (size_t j = 0; j < total; j++) {
                    min_val = std::min(min_val, data[j]);
                    max_val = std::max(max_val, data[j]);
                    sum += data[j];
                }
                std::cout << "  Min: " << min_val << ", Max: " << max_val 
                         << ", Mean: " << (sum / total) << std::endl;
                
                // Contar valores por encima de threshold
                int above_05 = 0, above_06 = 0, above_07 = 0;
                for (size_t j = 0; j < total; j++) {
                    if (data[j] > 0.5f) above_05++;
                    if (data[j] > 0.6f) above_06++;
                    if (data[j] > 0.7f) above_07++;
                }
                std::cout << "  Valores > 0.5: " << above_05 << std::endl;
                std::cout << "  Valores > 0.6: " << above_06 << std::endl;
                std::cout << "  Valores > 0.7: " << above_07 << std::endl;
            }
            
            // Cleanup
            for (auto name : input_names) free(const_cast<char*>(name));
            for (auto name : output_names) free(const_cast<char*>(name));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}