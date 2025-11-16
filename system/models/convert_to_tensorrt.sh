#g++ convert_onnx_to_trt.cpp -o convert_onnx_to_trt \
#    -std=c++17 \
#    -I/usr/src/tensorrt/include \
#    -I/usr/include/spdlog \
#    -L/usr/src/tensorrt/lib \
#    -lnvinfer \
#    -lnvonnxparser \
#    -lcuda \
#    -lcudart

#./convert_onnx_auto retinaface.onnx retinaface_fp16.engine

g++ convert_onnx_auto.cpp -o convert_onnx_auto \
    -std=c++17 \
    -I/usr/include/aarch64-linux-gnu \
    -I/usr/include/python3.8 \
    -L/usr/lib/aarch64-linux-gnu \
    -lnvinfer \
    -lnvonnxparser \
    -lnvinfer_plugin \
    -lpython3.8 \
    -lspdlog \
    -lfmt \
    -lpthread