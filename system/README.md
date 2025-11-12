# PANTO - Sistema de Reconocimiento Facial

Sistema modular de reconocimiento facial en tiempo real para NVIDIA Jetson Orin Nano.

## 🚀 Instalación Rápida

```bash
# 1. Instalar dependencias
sudo apt update
sudo apt install -y \
    libopencv-dev libsqlite3-dev libspdlog-dev \
    libgstreamer1.0-dev gstreamer1.0-plugins-{good,bad,ugly} \
    gstreamer1.0-libav cmake build-essential

# 2. Clonar repositorio
git clone https://github.com/tu-usuario/panto.git
cd panto/system

# 3. Descargar modelos ONNX
cd models
chmod +x setup_models.sh
./setup_models.sh
cd ..

# 4. Compilar
./build.sh

# 5. Probar detector
./run.sh test-img
```

## 📦 Modelos ONNX

El sistema usa modelos de [InsightFace buffalo_l](https://github.com/deepinsight/insightface/releases/tag/v0.7):

| Modelo | Origen | Función | Tamaño |
|--------|--------|---------|--------|
| **retinaface.onnx** | det_10g.onnx | Detección rostros | ~16 MB |
| **arcface_r100.onnx** | w600k_r50.onnx | Reconocimiento | ~250 MB |

### Descarga Automática

```bash
cd system/models
./setup_models.sh
```

El script descarga `buffalo_l.zip` oficial y extrae los modelos necesarios.

### Descarga Manual

Si falla la automática:

1. Descargar: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
2. Colocar en `system/models/buffalo_l.zip`
3. Ejecutar: `./setup_models.sh`

## 🎯 Uso

### Captura de Video RTSP

```bash
# Ver stream en vivo
./run.sh view-main       # Stream principal
./run.sh view-both       # Ambos streams

# Grabar
./run.sh record          # Grabar hasta Ctrl+C
./run.sh record 60       # Grabar 60 segundos
```

### Detección de Rostros

```bash
# Probar con imagen
./run.sh test-img                    # Usa test/img/test1.png
./run.sh test-img ruta/imagen.jpg    # Imagen personalizada

# Probar con video
./run.sh test-video                  # Usa último video grabado
./run.sh test-video videos/mi_video.mp4

# Ajustar threshold
./run.sh test-video videos/video.mp4 0.6   # Threshold 0.6
./run.sh test-video videos/video.mp4 0.4   # Más sensible
```

### Controles Interactivos (Video)

Durante la reproducción:
- **SPACE**: Pausar/reanudar
- **ESC**: Salir
- **S**: Guardar frame actual
- **+/-**: Ajustar threshold en tiempo real
- **F**: Modo rápido (sin display)

## ⚙️ Configuración

### Ajustar Threshold de Detección

El threshold controla cuán confiable debe ser una detección:

```bash
# En código (test/test_detector.cpp):
detector.set_conf_threshold(0.5f);  // 0.0-1.0, default 0.5

# Por línea de comandos:
./run.sh test-video video.mp4 0.7  # Más estricto (menos falsos positivos)
./run.sh test-video video.mp4 0.3  # Más permisivo (más detecciones)
```

**Recomendaciones**:
- **0.7-0.8**: Muy estricto, solo rostros muy claros
- **0.5-0.6**: Balanceado (recomendado)
- **0.3-0.4**: Permisivo, más falsos positivos

### Credenciales de Cámara

Editar `include/config.hpp`:

```cpp
constexpr const char* DEFAULT_USER = "admin";
constexpr const char* DEFAULT_PASS = "Panto2025";
constexpr const char* DEFAULT_IP = "192.168.0.101";
```

### Perfiles de Resolución

| Perfil | Resolución | FPS | Config | Uso |
|--------|-----------|-----|--------|-----|
| **720p** | 1280x720 | 20 | config_720p.toml | Bajo consumo |
| **1080p ROI** | 1920x1080 | 25 | config_1080p_roi.toml | Recomendado |
| **1080p Full** | 1920x1080 | 18 | config_1080p_full.toml | Alta precisión |
| **4K** | 3840x2160 | 20 | config_4k.toml | Máxima calidad |

```bash
./run.sh panto         # 1080p ROI (default)
./run.sh panto-720p    # 720p
./run.sh panto-4k      # 4K
```

## 🏗️ Arquitectura

```
system/
├── models/              # Modelos ONNX
│   ├── setup_models.sh  # Script de descarga
│   ├── retinaface.onnx  # Detector
│   └── arcface_r100.onnx # Reconocedor
├── configs/             # Configuraciones TOML
├── include/             # Headers C++
├── src/                 # Código fuente
│   ├── detector.cpp     # Detector de rostros
│   ├── stream_capture.cpp
│   └── utils.cpp
├── test/                # Programas de prueba
│   ├── test_detector.cpp       # Test con imagen/webcam
│   └── test_detector_video.cpp # Test con video MP4
├── build.sh             # Script de compilación
└── run.sh               # Comandos rápidos
```

### Librerías Modulares

```
libpanto_utils.so    - Utilidades base (GStreamer, retry)
libpanto_draw.so     - Visualización
libpanto_stream.so   - Captura RTSP
libpanto_detector.so - Detección facial
```

## 🐛 Troubleshooting

### No detecta rostros

```bash
# 1. Verificar que el modelo sea correcto
ls -lh models/retinaface.onnx  # Debe ser ~16 MB

# 2. Probar con threshold más bajo
./run.sh test-img ruta/imagen.jpg 0.3

# 3. Ver diagnóstico del modelo
./build/bin/diagnose_retinaface models/retinaface.onnx ruta/imagen.jpg
```

### Falsos positivos (detecta 2+ rostros en 1)

```bash
# Subir threshold
./run.sh test-video video.mp4 0.7

# O editar en código:
# test/test_detector.cpp línea ~25
detector.set_conf_threshold(0.7f);
```

### FPS bajo

```bash
# Usar perfil 720p
./run.sh panto-720p

# O verificar si CUDA está disponible:
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Debería mostrar: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Error al compilar

```bash
# Limpiar y recompilar
./clean.sh
./build.sh

# Verificar dependencias
sudo apt install libopencv-dev libspdlog-dev
```

## 📊 Performance

| Hardware | Resolución | FPS | Latencia |
|----------|-----------|-----|----------|
| **CPU** | 640x640 | 2-3 | ~400ms |
| **CUDA** | 640x640 | 25-30 | ~40ms |
| **TensorRT** | 640x640 | 60-80 | ~12ms |

Medido en Jetson Orin Nano.

## 🔧 Desarrollo

### Compilación Incremental

```bash
# Solo recompila archivos modificados
./build.sh

# Compilación limpia completa
./clean.sh
./build.sh
```

### Tests Individuales

```bash
# Detector con imagen
./build/bin/test_detector models/retinaface.onnx test/img/test1.png

# Detector con video
./build/bin/test_detector_video models/retinaface.onnx videos/video.mp4 0.5

# Diagnóstico low-level
./build/bin/diagnose_retinaface models/retinaface.onnx test/img/test1.png
```

## 📝 Comandos Completos (run.sh)

```bash
# Captura de video
./run.sh record           # Grabar main stream
./run.sh record-sub       # Grabar sub stream
./run.sh view-main        # Ver main
./run.sh view-both        # Ver ambos

# Detección
./run.sh test-img [imagen] [threshold]
./run.sh test-video [video] [threshold]

# Aplicación principal
./run.sh panto            # 1080p ROI
./run.sh panto-720p       # 720p
./run.sh panto-4k         # 4K
```
