# Face Recognition C++20 - Jetson Development Environment

Entorno completo de desarrollo para NVIDIA Jetson con OpenCV 4.10, CUDA, TensorRT, C++20 y Python.

## 🚀 Stack Tecnológico

- **C++20** con GCC 11
- **CMake 3.28**
- **OpenCV 4.10** con soporte CUDA
- **CUDA 11.4** + cuDNN
- **TensorRT 8.x**
- **Python 3.x** + JupyterLab
- **L4T JetPack r35.4.1**

## 📦 Setup Inicial

### 1. Clonar Repositorios

```bash
cd ~
mkdir -p ~/jetson_workspace
cd ~/jetson_workspace

# Clonar repositorios (reemplaza {} con tu token)
git clone https://asdcainicela:{TOKEN}@github.com/asdcainicela/lab-c-cpp.git 2>/dev/null || true
git clone https://asdcainicela:{TOKEN}@github.com/asdcainicela/panto_face_recognition_freh.git 2>/dev/null || true
```

### 2. Preparar y Construir

```bash
cd ~/jetson_workspace/panto_face_recognition_freh/docker
chmod +x run.sh docker-entrypoint.sh

# Construir imagen (tarda ~30-45 minutos)
docker build -t l4t-ml-cpp-py .

# Para build con logs detallados
DOCKER_BUILDKIT=0 docker build --no-cache -t l4t-ml-cpp-py . 2>&1 | tee build.log
```

### 3. Ejecutar Contenedor

```bash
./run.sh
```

## 🔍 Verificación del Sistema

### Dentro del Contenedor

```bash
# Verificación rápida
quick-check

# Verificación completa
cd /opt/tests
./verify_all.sh

# Tests individuales
./test_all           # Test C++ completo
python3 test_all.py  # Test Python completo
```

### Verificar Módulos Específicos

```bash
# OpenCV con CUDA
pkg-config --modversion opencv4
pkg-config --cflags opencv4
opencv_version --verbose

# CUDA devices
nvidia-smi
nvcc --version

# TensorRT
ls -lh /usr/lib/aarch64-linux-gnu/libnvinfer*
```

## 🐳 Gestión de Docker

### Información del Contenedor

```bash
# Ver contenedores activos
docker ps

# Ver todos los contenedores (incluyendo detenidos)
docker ps -a

# Ver imágenes
docker images

# Información detallada
docker inspect l4tmlcpppy
```

### Control del Contenedor

```bash
# Detener contenedor
docker stop l4tmlcpppy

# Iniciar contenedor detenido
docker start l4tmlcpppy

# Reiniciar contenedor
docker restart l4tmlcpppy

# Entrar al contenedor en ejecución
docker exec -it l4tmlcpppy bash

# Ver logs
docker logs l4tmlcpppy
docker logs -f l4tmlcpppy  # Follow mode
```

### Limpieza y Mantenimiento

```bash
# Eliminar contenedor específico (debe estar detenido)
docker rm l4tmlcpppy

# Forzar eliminación (aunque esté corriendo)
docker rm -f l4tmlcpppy

# Eliminar imagen
docker rmi l4t-ml-cpp-py

# Forzar eliminación de imagen
docker rmi -f l4t-ml-cpp-py

# ⚠️ LIMPIEZA AGRESIVA ⚠️

# Eliminar contenedores detenidos
docker container prune

# Eliminar imágenes sin usar
docker image prune

# Eliminar imágenes sin usar (incluyendo sin tags)
docker image prune -a

# Eliminar volúmenes no usados
docker volume prune

# Eliminar redes no usadas
docker network prune

# LIMPIEZA TOTAL (contenedores, redes, imágenes, cache)
docker system prune

# LIMPIEZA NUCLEAR (incluye volúmenes)
docker system prune -a --volumes

# Ver espacio usado
docker system df
```

### Rebuild desde Cero

```bash
# 1. Detener y eliminar contenedor
docker stop l4tmlcpppy
docker rm l4tmlcpppy

# 2. Eliminar imagen
docker rmi l4t-ml-cpp-py

# 3. Limpiar cache de build
docker builder prune -a

# 4. Rebuild
cd ~/jetson_workspace/panto_face_recognition_freh/docker
docker build -t l4t-ml-cpp-py .
```

## 📂 Estructura de Archivos

```
~/jetson_workspace/          → Montado en /workspace del contenedor
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

## 🖥️ Servicios Disponibles

### JupyterLab

```
URL: http://localhost:8888
Token: nvidia
```

Se inicia automáticamente al arrancar el contenedor. Logs en `/var/log/jupyter.log`

### Display X11

El contenedor tiene acceso al display del host para aplicaciones GUI:
- OpenCV `imshow()`
- Matplotlib visualizaciones
- Aplicaciones GTK

## 🛠️ Desarrollo C++

### Ejemplo de Compilación

```bash
# Con pkg-config
g++ -std=c++20 main.cpp -o app \
  $(pkg-config --cflags --libs opencv4) \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lcudart

# Con CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Template CMakeLists.txt

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

## 🐍 Desarrollo Python

### Paquetes Incluidos

- numpy, pandas, scipy
- matplotlib, seaborn
- scikit-learn
- pillow, tqdm
- opencv-python
- jupyterlab (con temas Catppuccin, Nord, Hale)

### Ejemplo OpenCV Python

```python
import cv2
import numpy as np

# Verificar CUDA
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")

# Usar GPU
img_gpu = cv2.cuda_GpuMat()
img_gpu.upload(img)
gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
result = gray_gpu.download()
```

## 🔧 Troubleshooting

### Problemas Comunes

**Error: "Cannot connect to the Docker daemon"**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# Logout y login de nuevo
```

**Error: "CUDA not found in OpenCV"**
```bash
# Verificar build de OpenCV
opencv_version --verbose | grep -i cuda
```

**JupyterLab no arranca**
```bash
docker exec -it l4tmlcpppy bash
cat /var/log/jupyter.log
jupyter lab --version
```

**Error X11 display**
```bash
xhost +local:docker
export DISPLAY=:0
```

**Contenedor no arranca después de reboot**
```bash
# El contenedor tiene --restart unless-stopped
docker start l4tmlcpppy

# Si no funciona, recrear
./run.sh
```

## 📊 Benchmarks Esperados

En Jetson Orin Nano (8GB):

- OpenCV CUDA `cvtColor` 1920x1080: ~2-3ms
- Matrix multiplication 1000x1000: ~15-20ms
- TensorRT inference MobileNetV2: ~5-8ms

## 🔗 Referencias

- [NVIDIA Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux)
- [OpenCV CUDA](https://docs.opencv.org/4.x/d1/d1a/tutorial_dnn_intro.html)
- [TensorRT Python](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

## 📝 Notas

- **Compilación**: OpenCV tarda ~25-35 minutos en compilar
- **Workspace**: Todo en `~/jetson_workspace` persiste entre reinicios
- **GPU Memory**: Monitor con `tegrastats` o `nvidia-smi`
- **Performance**: Usa `-j$(nproc)` para compilaciones paralelas

## 🤝 Contribuir

```bash
# Actualizar código
cd ~/jetson_workspace/panto_face_recognition_freh
git pull

# Rebuild si cambió Dockerfile
cd docker
docker build -t l4t-ml-cpp-py .
docker restart l4tmlcpppy
```

---

**Contacto**: gerald.cainicela.a@gmail.com  
**User**: userasd  
**Timezone**: America/Lima (GMT-5)