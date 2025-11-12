# PANTO - Real-Time Face Recognition System

Sistema de reconocimiento facial en tiempo real usando C++ para detección, tracking y almacenamiento en base de datos.

**Hardware Target:** NVIDIA Jetson Orin Nano (8GB RAM)

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Descarga de Modelos](#descarga-de-modelos)
- [Compilación](#compilación)
- [Configuración](#configuración)
- [Uso Rápido](#uso-rápido)
- [Perfiles de Configuración](#perfiles-de-configuración)
- [Base de Datos](#base-de-datos)
- [Troubleshooting](#troubleshooting)

---

## ✨ Características

- ✅ Detección de rostros con RetinaFace
- ✅ Tracking multi-objeto con ByteTrack
- ✅ Reconocimiento facial con ArcFace (embeddings 512D)
- ✅ Super-resolución condicional con Real-ESRGAN
- ✅ Grabación de video en tiempo real
- ✅ Almacenamiento en SQLite con embeddings
- ✅ Soporte para múltiples resoluciones (720p - 4K)
- ✅ Aceleración GPU con CUDA/TensorRT
- ✅ ROI (Region of Interest) configurable

---

## 🖥️ Requisitos del Sistema

### Hardware
- **Requerido:** NVIDIA Jetson Orin Nano (8GB RAM)
- **Recomendado:** 32GB+ almacenamiento SSD
- Cámara IP compatible con RTSP

### Software
- Ubuntu 20.04/22.04 (JetPack 5.x)
- CUDA 11.4+
- cuDNN 8.6+
- GStreamer 1.0
- CMake 3.10+
- C++17 compatible compiler

---

## 📦 Instalación

### 1. Dependencias del Sistema

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# OpenCV con CUDA
sudo apt install -y libopencv-dev

# SQLite3
sudo apt install -y libsqlite3-dev

# GStreamer
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# spdlog (logging)
sudo apt install -y libspdlog-dev

# TensorRT (ya incluido en JetPack)
# Verificar: dpkg -l | grep tensorrt
```

### 2. ONNX Runtime para Jetson

```bash
# Descargar ONNX Runtime optimizado para Jetson
cd ~/Downloads
wget https://nvidia.box.com/shared/static/pmsqsiaw4pg9qrbeckcbymho6c01jj4z.gz \
     -O onnxruntime-1.15.1-linux-aarch64.tar.gz

# Extraer
tar -xzf onnxruntime-1.15.1-linux-aarch64.tar.gz

# Mover archivos al sistema
sudo cp -r onnxruntime-linux-aarch64-1.15.1/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-aarch64-1.15.1/lib/* /usr/local/lib/

# Actualizar library cache
sudo ldconfig
```

### 3. Clonar Repositorio

```bash
cd ~
git clone https://github.com/tu-usuario/panto.git
cd panto
```

---

## 🤖 Descarga de Modelos

**Ver guía completa:** [MODELS.md](MODELS.md)

### Descarga Rápida

```bash
# Crear directorio de modelos
mkdir -p models
cd models

# 1. RetinaFace (detección de rostros)
wget https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx \
     -O retinaface.onnx

# 2. ArcFace R100 (reconocimiento facial)
wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx \
     -O arcface_r100.onnx

# 3. Real-ESRGAN x4 (super-resolución)
wget https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx \
     -O realesr_x4.onnx

cd ..
```

### Verificar Modelos

```bash
ls -lh models/
# Deberías ver:
# retinaface.onnx       (~27 MB)
# arcface_r100.onnx     (~250 MB)
# realesr_x4.onnx       (~67 MB)
```

---

## 🔨 Compilación

### Compilación Rápida

```bash
# Hacer ejecutables los scripts
chmod +x build.sh clean.sh run.sh

# Compilar proyecto
./build.sh

# Opcional: compilar en modo Debug
./build.sh Debug
```

### Compilación Manual

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..
```

### Salida Esperada

```
build/
├── bin/
│   ├── record          # Grabación sin display
│   ├── view            # Visualización sin grabación
│   ├── record_view     # Grabación + visualización
│   └── panto           # Aplicación principal
└── lib/
    ├── libpanto_utils.so
    ├── libpanto_draw.so
    └── libpanto_stream.so
```

---

## ⚙️ Configuración

### Configuración de Cámara

Editar `config.toml`:

```toml
[camera]
device_id = 0
backend = "gstreamer"
```

Editar credenciales en código (por ahora):
- **Usuario:** `admin`
- **Contraseña:** `Panto2025`
- **IP:** `192.168.0.101`
- **Puerto:** `554`

### Estructura de Directorios

```bash
# Crear directorios necesarios
mkdir -p data/captures videos logs
```

---

## 🚀 Uso Rápido

### Script run.sh

```bash
# Ver opciones disponibles
./run.sh

# Grabar stream principal (hasta Ctrl+C)
./run.sh 1

# Grabar 60 segundos
./run.sh 1 60

# Ver ambos streams (main + sub)
./run.sh 3

# Ver solo main stream
./run.sh 4

# Grabar + visualizar 30 segundos
./run.sh 6 30

# Ejecutar PANTO principal (1080p ROI)
./run.sh 8

# PANTO en 720p
./run.sh 9

# PANTO en 4K
./run.sh 10
```

### Ejecución Manual

```bash
# Solo grabar (headless)
./build/bin/record main          # Hasta Ctrl+C
./build/bin/record main 60       # 60 segundos
./build/bin/record sub 120       # Sub stream 120s

# Solo visualizar
./build/bin/view                 # Ambos streams
./build/bin/view main            # Solo main
./build/bin/view sub             # Solo sub

# Grabar + visualizar
./build/bin/record_view main
./build/bin/record_view sub 60

# PANTO principal
./build/bin/panto --config configs/config_1080p_roi.toml
```

---

## 📊 Perfiles de Configuración

| Perfil | Resolución | FPS | Precisión | ROI | SR |
|--------|-----------|-----|-----------|-----|----|
| `config_4k.toml` | 3840x2160 | 18-22 | 97-99% | No | Raro |
| `config_1440p.toml` | 2560x1440 | 20-24 | 96-98% | No | Ocasional |
| `config_1080p_roi.toml` | 1920x1080 | 22-25 | 94-97% | Sí | Condicional |
| `config_1080p_full.toml` | 1920x1080 | 15-18 | 90-94% | No | Frecuente |
| `config_720p.toml` | 1280x720 | 18-22 | 86-90% | Sí | Casi siempre |

**✅ Recomendado:** `config_1080p_roi.toml` para Jetson Orin Nano

### Ejemplo de Uso con Perfil

```bash
./build/bin/panto --config configs/config_720p.toml
```

---

## 💾 Base de Datos

### Esquema SQLite

```sql
-- Detecciones en tiempo real
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    face_embedding BLOB NOT NULL,
    confidence REAL,
    face_size INTEGER,
    image_path TEXT,
    track_id INTEGER
);

-- Rostros conocidos
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT UNIQUE,
    embedding BLOB NOT NULL,
    captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT
);

CREATE INDEX idx_timestamp ON detections(timestamp);
CREATE INDEX idx_track_id ON detections(track_id);
```

### Ubicación de Datos

- Base de datos: `data/faces.db`
- Imágenes capturadas: `data/captures/YYYY-MM-DD/`
- Videos: `videos/recording_*.mp4`
- Logs: `logs/panto.log`

### Consultas Útiles

```bash
# Ver últimas 10 detecciones
sqlite3 data/faces.db \
  "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;"

# Contar detecciones por día
sqlite3 data/faces.db \
  "SELECT DATE(timestamp), COUNT(*) FROM detections GROUP BY DATE(timestamp);"
```

---

## 🐛 Troubleshooting

### FPS Bajo (<15)

```bash
# 1. Desactivar super-resolución
# En config: superresolution.enabled = false

# 2. Usar perfil más bajo
./build/bin/panto --config configs/config_720p.toml

# 3. Reducir ROI
# En config: roi.width=640, roi.height=360
```

### No Detecta Rostros

```bash
# 1. Verificar modelos
ls -lh models/
# Deben existir los 3 archivos .onnx

# 2. Calibrar ROI (pendiente implementar)
# ./build/bin/panto --calibrate-roi

# 3. Bajar threshold
# En config: confidence_threshold = 0.5
```

### Error de Conexión RTSP

```bash
# Verificar cámara
ping 192.168.0.101

# Probar URL RTSP manualmente
gst-launch-1.0 rtspsrc location=rtsp://admin:Panto2025@192.168.0.101:554/main ! fakesink

# Revisar logs
tail -f logs/panto.log
```

### Error de CUDA/TensorRT

```bash
# Verificar CUDA
nvcc --version

# Verificar TensorRT
dpkg -l | grep tensorrt

# Deshabilitar aceleración GPU temporalmente
# En config: use_tensorrt = false, use_cuda = false
```

### Compilación Fallida

```bash
# Limpiar build
./clean.sh

# Verificar dependencias
pkg-config --modversion opencv4
pkg-config --modversion spdlog

# Recompilar con verbose
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON ..
make -j$(nproc)
```

---

## 📝 Logs

### Ver Logs en Tiempo Real

```bash
tail -f logs/panto.log
```

### Niveles de Log

Editar en `config.toml`:

```toml
[logging]
level = "INFO"  # DEBUG, INFO, WARN, ERROR
log_to_file = true
log_to_console = true
```

---

## 🗂️ Estructura del Proyecto

```
panto/
├── build/              # Compilados (generado)
├── configs/            # Perfiles de configuración
├── data/               # Base de datos y capturas
├── include/            # Headers C++
├── logs/               # Archivos de log
├── models/             # Modelos ONNX (descargar)
├── src/                # Código fuente
├── test/               # Programas de prueba
├── videos/             # Grabaciones (generado)
├── CMakeLists.txt      # Configuración CMake
├── config.toml         # Configuración principal
├── build.sh            # Script de compilación
├── run.sh              # Script de ejecución rápida
└── README.md           # Esta documentación
```

---

## 📚 Documentación Adicional

- **[MODELS.md](MODELS.md)** - Guía completa de modelos ONNX
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Arquitectura del sistema

---

## 🔄 Flujo de Trabajo Típico

```bash
# 1. Primera vez: descargar modelos
cd models && ./download_models.sh

# 2. Compilar
./build.sh

# 3. Probar captura básica
./run.sh 4  # Ver main stream

# 4. Ejecutar sistema completo
./run.sh 8  # PANTO con config 1080p ROI
```
