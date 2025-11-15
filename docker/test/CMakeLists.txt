# Compilar con CUDA headers
g++ test_opencv_cuda_tensorrt.cpp -o test_opencv_cuda_tensorrt \
  -I/usr/local/cuda/include \
  -I/usr/include/aarch64-linux-gnu \
  $(pkg-config --cflags --libs opencv4) \
  -L/usr/local/cuda/lib64 \
  -L/usr/lib/aarch64-linux-gnu \
  -lnvinfer \
  -lcudart \
  -std=c++17

# Ejecutar
./test_opencv_cuda_tensorrt