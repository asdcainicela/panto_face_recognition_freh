g++ convert_onnx_to_trt.cpp -o convert_onnx_to_trt \
    -std=c++17 \
    -I/usr/src/tensorrt/include \
    -I/usr/include/spdlog \
    -L/usr/src/tensorrt/lib \
    -lnvinfer \
    -lnvonnxparser \
    -lcuda \
    -lcudart
