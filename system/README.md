/*
================================================================================
PANTO - MODELOS ONNX ACTUALIZADOS (2025)
================================================================================

# NUEVOS ENLACES DE DESCARGA

1. RECONOCIMIENTO FACIAL - ArcFace R100
   - Hugging Face: https://huggingface.co/garavv/arcface-onnx
   - Descarga directa:
     wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx?download=true -O models/arcface_r100.onnx

2. SUPER-RESOLUCIÓN - Real-ESRGAN x4plus
   - Hugging Face: https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/blob/01179a4da7bf5ac91faca650e6afbf282ac93933/Real-ESRGAN-x4plus.onnx
   - Descarga directa:
     wget https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx -O models/realesr_x4.onnx

3. DETECCIÓN DE ROSTROS - RetinaFace (ResNet50)
   - Hugging Face: https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/blob/main/retinaface-resnet50.onnx
   - Descarga directa:
     wget https://huggingface.co/TheEeeeLin/HivisionIDPhotos_matting/resolve/main/retinaface-resnet50.onnx -O models/retinaface.onnx

   Alternativa de exportación:
   - GitHub: https://github.com/kingardor/retinaface-onnx
   - Script de conversión disponible: export_onnx.py (desde .pth → .onnx)


================================================================================
NOTAS DE INTEGRACIÓN
================================================================================

1. Directorio de destino
   Coloca todos los modelos descargados en:
       panto/models/
   y verifica que los nombres coincidan con los declarados en config.toml:
       retinaface.onnx
       arcface_r100.onnx
       realesr_x4.onnx

2. Compatibilidad de entrada/salida
   - ArcFace: entrada 112x112 RGB, salida embedding de 512 dimensiones (float32)
   - RetinaFace: entrada 640x640 o similar, salida con bounding boxes y landmarks
   - Real-ESRGAN: entrada variable, idealmente múltiplos de 64 (x4 upscaling)

3. Aceleración en Jetson Orin Nano
   - Convierte los modelos ONNX a TensorRT para mayor rendimiento:
         trtexec --onnx=models/arcface_r100.onnx --saveEngine=models/arcface_r100.trt
         trtexec --onnx=models/retinaface.onnx --saveEngine=models/retinaface.trt
   - En config.toml:
         [performance]
         use_tensorrt = true

4. Recomendaciones
   - Verifica licencias antes de uso comercial.
   - Usa ONNX Runtime o TensorRT según disponibilidad de CUDA/cuDNN.
   - Comprueba la coherencia de dimensiones antes del embedding.
   - Guarda tus modelos en caché local (no dependas de descargas en tiempo de ejecución).

================================================================================
*/


# PANTO - Reconocimiento Facial en C++

Sistema de reconocimiento facial en tiempo real usando C++ para deteccion, tracking y almacenamiento de rostros en base de datos.

**Hardware:** NVIDIA Jetson Orin Nano (8GB RAM)

---

## Estructura del Proyecto

```
panto/
├── config.toml
├── configs/
│   ├── config_4k.toml
│   ├── config_1440p.toml
│   ├── config_1080p_roi.toml
│   ├── config_1080p_full.toml
│   └── config_720p.toml
├── models/
│   ├── retinaface.onnx
│   ├── arcface_r100.onnx
│   └── realesr_x2.onnx
├── data/
│   ├── faces.db
│   └── captures/
├── include/
│   ├── utils.hpp
│   └── stream_capture.hpp
├── src/
│   ├── utils.cpp
│   └── stream_capture.cpp
│   ├── detector.cpp
│   ├── tracker.cpp
│   ├── recognizer.cpp
│   └── database.cpp
├── test/
│   ├── record.cpp
│   ├── view.cpp
│   └── record_view.cpp
├── build/
│   ├── bin/
│   │   ├── record
│   │   ├── view
│   │   ├── record_view
│   │   └── panto
│   └── lib/
│       └── libstream_lib.so
├── CMakeLists.txt
├── build.sh
└── run.sh
└── main.cpp
```

---

## Perfiles de Configuracion

| Resolucion | Archivo | FPS | Precision | ROI | SR |
|------------|---------|-----|-----------|-----|----|
| 4K (3840x2160) | `config_4k.toml` | 18-22 | 97-99% | Opcional | Raro |
| 1440p (2560x1440) | `config_1440p.toml` | 20-24 | 96-98% | Opcional | Ocasional |
| 1080p + ROI | `config_1080p_roi.toml` | 22-25 | 94-97% | Si | Condicional |
| 1080p Full | `config_1080p_full.toml` | 15-18 | 90-94% | No | Frecuente |
| 720p (1280x720) | `config_720p.toml` | 18-22 | 86-90% | Si | Casi siempre |

**Recomendado para Jetson Orin Nano:** `config_1080p_roi.toml`

---

## Modelos Necesarios

Todos los modelos son pre-entrenados. NO necesitas entrenar nada.

### 1. Deteccion de Rostros - RetinaFace
- Formato: ONNX
- URL: https://github.com/onnx/models/tree/main/vision/body_analysis/retinaface
- Archivo: `retinaface_mobilenet.onnx`

### 2. Reconocimiento - ArcFace R100
- Formato: ONNX
- URL: https://github.com/deepinsight/insightface/tree/master/model_zoo
- Archivo: `arcface_r100.onnx`
- Embeddings: 512 dimensiones

### 3. Super-Resolution - RealESR-General
- Formato: ONNX
- URL: https://github.com/xinntao/Real-ESRGAN
- Archivo: `realesr_x2.onnx` o `realesr_x4.onnx`

### 4. Tracking - ByteTrack
- No requiere modelo
- Implementado en codigo
- Referencia: https://github.com/ifzhang/ByteTrack

---

## Dependencias

```bash
# OpenCV con CUDA
sudo apt install libopencv-dev

# ONNX Runtime para Jetson
wget https://nvidia.box.com/shared/static/...onnxruntime.tar.gz
tar -xzvf onnxruntime.tar.gz

# SQLite3
sudo apt install libsqlite3-dev

# GStreamer
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

---

## Compilacion

```bash
chmod +x build.sh
./build.sh
```

Esto genera:
- Libreria compartida: `build/lib/libstream_lib.so`
- Ejecutables de prueba: `build/bin/record`, `build/bin/view`, `build/bin/record_view`
- Aplicacion principal: `build/bin/panto`

---

## Ejecucion Rapida

### Usando script run.sh

```bash
chmod +x run.sh

# Grabar main stream hasta Ctrl+C
./run.sh 1

# Grabar main 60 segundos
./run.sh 1 60

# Ver ambos streams
./run.sh 3

# Ver solo main
./run.sh 4

# Grabar + ver main 30 segundos
./run.sh 6 30

# Ejecutar PANTO principal
./run.sh 8
```

### Ejecucion Manual

```bash
# Solo grabar (sin visualizacion)
./build/bin/record main          # Hasta Ctrl+C
./build/bin/record main 60       # 60 segundos
./build/bin/record sub 120       # Sub stream 120s

# Solo visualizar
./build/bin/view                 # Ambos streams
./build/bin/view main            # Solo main
./build/bin/view sub             # Solo sub

# Grabar + visualizar
./build/bin/record_view main     # Hasta Ctrl+C
./build/bin/record_view sub 60   # Sub 60 segundos

# Aplicacion principal PANTO
./build/bin/panto --config configs/config_1080p_roi.toml
```

---

## Base de Datos

### Esquema SQLite

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    face_embedding BLOB NOT NULL,
    confidence REAL,
    face_size INTEGER,
    image_path TEXT,
    track_id INTEGER
);

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

### Almacenamiento
- Base de datos: `data/faces.db`
- Imagenes: `data/captures/{date}/{timestamp}.jpg`
- Embeddings: Vector de 512 floats (2048 bytes)

---

## Configuracion Principal

Archivo: `config.toml`

```toml
[system]
profile = "configs/config_1080p_roi.toml"
models_path = "models/"
data_path = "data/"

[camera]
device_id = 0
backend = "gstreamer"

[database]
path = "data/faces.db"
save_images = true
images_path = "data/captures/"

[performance]
use_tensorrt = true
num_threads = 4
```

---

## Arquitectura del Sistema

```
Frame Input (Camera)
         |
         v
    Apply ROI (opcional)
         |
         v
    Face Detection (RetinaFace)
         |
         v
    Tracking (ByteTrack)
         |
         v
    Super-Resolution (condicional)
         |
         v
    Face Recognition (ArcFace)
         |
         v
    Database Storage (SQLite)
```

---

## Calibracion Inicial

### 1. Verificar ROI
```bash
./panto --config configs/config_1080p_roi.toml --calibrate-roi
```

### 2. Medir Tamanos de Rostro
```bash
./panto --config configs/config_1080p_roi.toml --measure-faces
```

### 3. Prueba de Performance
```bash
./panto --config configs/config_1080p_roi.toml --benchmark
```

---

## Troubleshooting

### FPS Bajo (<15)
1. Desactivar SR: `superresolution.enabled = false`
2. Reducir ROI: `width=640, height=360`
3. Usar perfil mas bajo: `config_720p.toml`

### No Detecta Rostros
1. Verificar ROI con: `--calibrate-roi`
2. Bajar threshold: `detection.confidence_threshold = 0.5`
3. Reducir min size: `detection.min_face_size = 40`

### Errores de Base de Datos
1. Verificar permisos: `chmod 666 data/faces.db`
2. Recrear schema: `sqlite3 data/faces.db < schema.sql`

---

## Logs

```bash
# Ver logs en tiempo real
tail -f logs/panto.log

# Consultar base de datos
sqlite3 data/faces.db "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;"
```

---

## Licencia

MIT License