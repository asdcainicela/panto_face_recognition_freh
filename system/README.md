cd /workspace/panto_face_recognition_freh/system

# 1. Recompilar con detector
./build.sh

# 2. Probar captura con pipeline mejorado
./run.sh 4

# 3. Test detector standalone (webcam)
./build/bin/test_detector models/retinaface.onnx

# 4. Test detector con imagen
./build/bin/test_detector models/retinaface.onnx ruta/a/imagen.jpg


# PANTO - Sistema de Reconocimiento Facial

Sistema modular de reconocimiento facial en tiempo real para NVIDIA Jetson Orin Nano.

## Requisitos

- Ubuntu 20.04/22.04 (JetPack 5.x)
- CUDA 11.4+, cuDNN 8.6+
- OpenCV con CUDA
- GStreamer 1.0
- SQLite3, spdlog

## Instalación Rápida

```bash
# 1. Instalar dependencias
sudo apt update
sudo apt install -y libopencv-dev libsqlite3-dev libspdlog-dev \
    libgstreamer1.0-dev gstreamer1.0-plugins-{good,bad,ugly} \
    gstreamer1.0-libav

# 2. Clonar repositorio
git clone https://github.com/tu-usuario/panto.git
cd panto

# 3. Descargar modelos (EJECUTAR DESDE RAIZ)
cd models
chmod +x download_models.sh
./download_models.sh
cd ..

# 4. Compilar
chmod +x build.sh run.sh
./build.sh

# 5. Probar captura básica
./run.sh 4
```

## Uso

```bash
./run.sh 1        # Grabar main stream
./run.sh 3        # Ver ambos streams
./run.sh 8        # Ejecutar PANTO (1080p ROI)
./run.sh 9        # PANTO en 720p
```

## Configuración de Cámara

Editar credenciales en `include/config.hpp`:
- Usuario: `admin`
- Password: `Panto2025`
- IP: `192.168.0.101`

## Perfiles Disponibles

| Perfil | Resolución | FPS | Uso |
|--------|-----------|-----|-----|
| 720p | 1280x720 | 20 | Bajo consumo |
| 1080p ROI | 1920x1080 | 25 | **Recomendado** |
| 1080p Full | 1920x1080 | 18 | Alta precisión |
| 1440p | 2560x1440 | 22 | Muy alta precisión |
| 4K | 3840x2160 | 20 | Máxima calidad |

## Estructura

```
panto/
├── build/          # Compilados
├── configs/        # TOML configs
├── models/         # Modelos ONNX
├── include/        # Headers
├── src/            # Código fuente
├── test/           # Tests
└── videos/         # Grabaciones
```

## Troubleshooting

**FPS bajo:** Usar `config_720p.toml` o desactivar super-resolución

**No conecta:** Verificar IP con `ping 192.168.0.101`

**Error CUDA:** Revisar `nvcc --version` y `dpkg -l | grep tensorrt`

## Base de Datos

- SQLite: `data/faces.db`
- Capturas: `data/captures/YYYY-MM-DD/`
- Videos: `videos/recording_*.mp4`