#torch.onnx.export(
#    model, dummy, "buffalo_l.onnx",
#    input_names=["input_1"],
#    dynamic_axes={"input_1": {0: "batch"}, "embedding": {0: "batch"}},
#    ...
#)

#dynamic_axes={
#    "input":  {0: "batch", 2: "height", 3: "width"},
#    "bbox":   {0: "batch", 1: "num_anchors"},
#    ...
#}

g++ convert_onnx_auto.cpp -o convert_onnx_auto \
    -std=c++17 \
    -O3 \
    -I/usr/local/cuda/include \
    -I/usr/include/aarch64-linux-gnu \
    -I/usr/include/python3.8 \
    -L/usr/local/cuda/lib64 \
    -L/usr/lib/aarch64-linux-gnu \
    -lnvinfer \
    -lnvonnxparser \
    -lnvinfer_plugin \
    -lcudart \
    -lpython3.8 \
    -lspdlog \
    -lpthread \
    -Wl,-rpath,/usr/local/cuda/lib64 \
    -Wl,-rpath,/usr/lib/aarch64-linux-gnu