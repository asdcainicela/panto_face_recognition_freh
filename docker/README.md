# Face Recognition C++20 – Jetson Development Environment

Entorno completo de desarrollo para NVIDIA Jetson con OpenCV 4.10, CUDA, TensorRT, C++20 y Python.

## Stack Tecnológico

* C++20 con GCC 11
* CMake 3.28
* OpenCV 4.10 con soporte CUDA
* CUDA 11.4 + cuDNN
* TensorRT 8.x
* Python 3.x + JupyterLab
* L4T JetPack r35.4.1

## Setup Inicial

### 1. Clonar repositorios

```bash
cd ~
mkdir -p ~/jetson_workspace
cd ~/jetson_workspace

git clone https://asdcainicela:{TOKEN}@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || true
git clone https://asdcainicela:{TOKEN}@github.com/asdcainicela/panto_face_recognition_freh.git 2>/dev/null || true
```

### 2. Construir Docker

```bash
cd ~/jetson_workspace/panto_face_recognition_freh/docker
chmod +x run.sh docker-entrypoint.sh

docker build -t l4t-ml-cpp-py .

DOCKER_BUILDKIT=0 docker build --no-cache -t l4t-ml-cpp-py . 2>&1 | tee build.log
```

### 3. Ejecutar contenedor

```bash
./run.sh
```

## Verificación del sistema dentro del contenedor

```bash
quick-check

cd /opt/tests
./verify_all.sh

./test_all
python3 test_all.py
```

### Verificar módulos

```bash
pkg-config --modversion opencv4
pkg-config --cflags opencv4
opencv_version --verbose

nvidia-smi
nvcc --version

ls -lh /usr/lib/aarch64-linux-gnu/libnvinfer*
```

## Gestión de Docker

### Información del contenedor

```bash
docker ps
docker ps -a
docker images
docker inspect l4tmlcpppy
```

### Control del contenedor

```bash
docker stop l4tmlcpppy
docker start l4tmlcpppy
docker restart l4tmlcpppy
docker exec -it l4tmlcpppy bash
docker logs l4tmlcpppy
docker logs -f l4tmlcpppy
```

### Limpieza

```bash
docker rm l4tmlcpppy
docker rm -f l4tmlcpppy

docker rmi l4t-ml-cpp-py
docker rmi -f l4t-ml-cpp-py

docker container prune
docker image prune
docker image prune -a
docker volume prune
docker network prune

docker system prune
docker system prune -a --volumes

docker system df
```

### Rebuild desde cero

```bash
docker stop l4tmlcpppy
docker rm l4tmlcpppy
docker rmi l4t-ml-cpp-py
docker builder prune -a

cd ~/jetson_workspace/panto_face_recognition_freh/docker
docker build -t l4t-ml-cpp-py .
```

## Estructura de archivos

```
~/jetson_workspace/
├── lab-c-cpp/
├── panto_face_recognition_freh/
│   └── docker/
│       ├── Dockerfile
│       ├── docker-entrypoint.sh
│       ├── run.sh
│       ├── README.md
│       └── test/
│           ├── CMakeLists.txt
│           ├── test_all.cpp
│           ├── test_all.py
│           └── verify_all.sh
```

## Servicios disponibles

### JupyterLab

```
URL: http://localhost:8888
Token: nvidia
```

Logs: `/var/log/jupyter.log`

### Acceso X11

Habilitado para imshow y aplicaciones GUI.

## Desarrollo C++

### Compilación con pkg-config

```bash
g++ -std=c++20 main.cpp -o app \
  $(pkg-config --cflags --libs opencv4) \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lcudart
```

### Compilación con CMake

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### CMakeLists.txt de referencia

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyProject CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

add_executable(app main.cpp)

target_include_directories(app PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(app
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
)
```

## Desarrollo Python

Paquetes incluidos: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, pillow, tqdm, opencv-python, jupyterlab.

### Ejemplo

```python
import cv2

print(cv2.cuda.getCudaEnabledDeviceCount())

img_gpu = cv2.cuda_GpuMat()
img_gpu.upload(img)
gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
result = gray_gpu.download()
```

## Troubleshooting

### Docker daemon

```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### OpenCV sin CUDA

```bash
opencv_version --verbose | grep -i cuda
```

### JupyterLab no inicia

```bash
cat /var/log/jupyter.log
```

### Problemas X11

```bash
xhost +local:docker
export DISPLAY=:0
```

## Benchmarks esperados (Jetson Orin Nano 8GB)

* cvtColor 1080p CUDA: 2–3 ms
* Matmul 1000x1000: 15–20 ms
* TensorRT MobileNetV2: 5–8 ms

## Contribuir

```bash
cd ~/jetson_workspace/panto_face_recognition_freh
git pull

cd docker
docker build -t l4t-ml-cpp-py .
docker restart l4tmlcpppy
```

## Contacto

[gerald.cainicela.a@gmail.com](mailto:gerald.cainicela.a@gmail.com)
userasd
America/Lima (GMT-5)
