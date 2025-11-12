# PANTO - Arquitectura del Sistema

Documentación técnica de arquitectura modular y flujo de datos.

## Visión General

Sistema modular de reconocimiento facial en tiempo real para dispositivos edge.

Principios:
- Separación de responsabilidades
- Bajo acoplamiento entre módulos
- Alta cohesión interna
- Compilación incremental

## Arquitectura Modular

### Librerías Compartidas

### Librerías Compartidas

```
libpanto_utils.so    - Utilidades base (pipeline GStreamer, retry)
libpanto_draw.so     - Funciones de visualización
libpanto_stream.so   - Captura y grabación RTSP
```

Ventajas:
- Compilación incremental rápida
- Reutilización entre ejecutables
- Testing modular independiente
- Binarios más pequeños

## Flujo de Procesamiento

### Pipeline Video

```
RTSP Camera → GStreamer → StreamCapture → Recording/Display
```

Detalles:
1. Cámara IP transmite RTSP
2. GStreamer decodifica H.264 con nvv4l2decoder
3. StreamCapture lee frames y actualiza stats
4. Bifurcación: grabación MP4 y/o visualización

### Pipeline Reconocimiento (Futuro)

```
Frame → ROI → Detection → Tracking → SR (condicional) → Recognition → DB
```

Pasos:
1. Aplicar ROI opcional
2. RetinaFace detecta rostros
3. ByteTrack asigna IDs
4. Real-ESRGAN solo si rostro pequeño
5. ArcFace genera embedding 512D
6. SQLite busca match o guarda nuevo

## Componentes

### StreamCapture

Gestión de captura RTSP.
- Reconexión automática en caso de pérdida
- Estadísticas en tiempo real (FPS, frames perdidos)
- Grabación opcional sin impactar visualización
- Control mediante señales (Ctrl+C)

---

### 2. DrawUtils

**Responsabilidad:** Renderizado de información sobre frames.

```cpp
namespace DrawUtils {
    struct DrawConfig {
        bool show_border;
        bool show_stream_name;
        bool show_fps;
        bool show_recording;
        cv::Scalar color;
    };
    
    void draw_stream_info(cv::Mat& frame, const StreamStats& stats, ...);
    void draw_recording_indicator(cv::Mat& frame, bool is_recording, ...);
    void draw_fps_counter(cv::Mat& frame, double fps, ...);
}
```

**Características:**
- Configuración flexible de overlay
- No modifica lógica de captura
- Reutilizable en múltiples contextos

---

### 3. Config

**Responsabilidad:** Constantes y configuración centralizada.

```cpp
namespace Config {
    // Camera defaults
    constexpr const char* DEFAULT_USER = "admin";
    constexpr const char* DEFAULT_PASS = "Panto2025";
    constexpr const char* DEFAULT_IP = "192.168.0.101";
    
    // Resolution profiles
    struct Resolution {
        int width, height;
        std::string config_file;
    };
    
    std::string get_config_for_resolution(const cv::Size& size);
}
```

**Características:**
- Sin magic numbers en código
- Fácil cambio de defaults
- Detección automática de resolución

---

### 4. Utils

**Responsabilidad:** Utilidades compartidas.

```cpp
std::string gst_pipeline(const std::string& user, 
                        const std::string& pass, 
                        const std::string& ip, 
                        int port, 
                        const std::string& stream_type);

cv::VideoCapture open_cap(const std::string& pipeline, int retries);
```

**Características:**
- Construcción de GStreamer pipeline
- Retry logic con backoff
- Logging integrado

---

## 🎭 Pipeline de Reconocimiento

### Diagrama de Estados

```
                    ┌──────────────┐
                    │   DETECT     │
                    │  (Far Zone)  │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │   TRACK      │
                    │ (Approaching)│
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
            ┌──────►│ ACCUMULATE   │◄──────┐
            │       │ (Multi-frame)│       │
            │       └──────┬───────┘       │
            │              │                │
            │              v                │
            │       ┌──────────────┐       │
            │       │   OPTIMAL    │       │
Low         │       │    ZONE      │       │ High
Confidence  │       │  (Decision)  │       │ Confidence
            │       └──────┬───────┘       │
            │              │                │
            │              v                │
            │       ┌──────────────┐       │
            └───────┤   MATCH?     ├───────┘
                    │  Known/New   │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │   VERIFY     │
                    │ (Close Zone) │
                    └──────────────┘
```

### Zonas de Procesamiento

| Zona | Tamaño Rostro | Acción | SR | Peso |
|------|--------------|--------|----|----|
| **Far** | 60-100px | Track + Accumulate | ✅ | 0.3 |
| **Optimal** | 100-200px | Definitive Match | ❌ | 1.0 |
| **Close** | 200+px | Verify + Log | ❌ | 1.0 |

**Estrategia:**
- **Far Zone:** Aplicar SR, comenzar acumulación
- **Optimal Zone:** Tomar decisión final (mejor balance calidad/costo)
- **Close Zone:** Verificación de confianza

---

## 🧠 Decisiones de Diseño

### Separación Display vs Capture

**Problema:** Mezclar captura y visualización complica testing y reutilización.

**Solución:**
```cpp
// ANTES (acoplado)
class StreamViewer {
    void run() {
        while (read(frame)) {
            putText(...);  // Display mezclado
            imshow(...);
            writer.write(frame);  // Recording mezclado
        }
    }
};

// DESPUÉS (desacoplado)
class StreamCapture {
    void run() {
        while (read(frame)) {
            if (recording_enabled) writer.write(frame);
            if (viewing_enabled) show_frame(frame);
        }
    }
};
```

**Beneficios:**
- Headless recording (sin X11)
- Testing sin display
- Diferentes visualizaciones sin tocar captura

---

### Config Centralizado

**Problema:** Magic numbers repetidos en todo el código.

**Solución:**
```cpp
// ANTES
cv::Point(10, 20)  // Repetido 50 veces
cv::Scalar(255, 0, 0)  // Repetido 30 veces

// DESPUÉS
Config::margin_x
Config::margin_y
Config::default_color
```

**Beneficios:**
- Cambio único para todos los usos
- Autocompletado IDE
- Documentación en un solo lugar

---

### Modularidad de Librerías

**Problema:** Recompilar todo por cambio pequeño.

**Solución:**
```
libpanto_utils.so    (base - raramente cambia)
libpanto_draw.so     (cambios frecuentes de UI)
libpanto_stream.so   (lógica core)
```

**Beneficios:**
- Compilación incremental rápida
- Shared libraries más pequeñas
- Fácil profiling de módulos

---

## ⚡ Optimizaciones

### 1. ROI (Region of Interest)

**Beneficio:** Procesar solo área relevante → reduce cómputo 50-70%.

```toml
[camera.roi]
enabled = true
x = 480
y = 270
width = 960   # Solo 25% del frame
height = 540
```

**Casos de uso:**
- Cámara fija con área de tránsito conocida
- Entrada/puerta específica

---

### 2. Super-Resolución Condicional

**Beneficio:** Aplicar SR solo cuando es necesario.

```cpp
if (face_size < threshold) {
    face = apply_superresolution(face);  // Solo rostros pequeños
}
```

**Ahorro:** ~40% de cómputo en escenarios típicos.

---

### 3. Multi-Frame Voting

**Beneficio:** Mejorar confianza acumulando múltiples frames.

```cpp
Track track;
for (int i = 0; i < min_frames; i++) {
    embedding = recognize(track.frames[i]);
    track.embeddings.push(embedding);
}
final_decision = vote(track.embeddings);
```

**Ventajas:**
- Reduce falsos positivos
- Mayor robustez ante oclusiones

---

### 4. TensorRT Optimization

**Beneficio:** 2-3x speedup en Jetson.

```bash
# Conversión ONNX → TensorRT
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --workspace=2048
```

### TensorRT

Conversión ONNX a TensorRT da 2-3x speedup.

Optimizaciones:
- FP16 precision (vs FP32)
- Kernel fusion
- Memory optimization
- Workspace tuning

## Métricas Performance

Targets por resolución:

```
720p + ROI:  20-22 FPS, <50ms latencia, ~2GB RAM, 86-90% precisión
1080p + ROI: 22-25 FPS, <50ms latencia, ~3GB RAM, 94-97% precisión
1080p Full:  15-18 FPS, ~70ms latencia, ~4GB RAM, 90-94% precisión
1440p:       20-24 FPS, ~60ms latencia, ~5GB RAM, 96-98% precisión
4K:          18-22 FPS, ~80ms latencia, ~6GB RAM, 97-99% precisión
```

## Roadmap

Fase 1 - Infraestructura (Actual):
- Captura RTSP estable
- Recording/viewing desacoplados
- Arquitectura modular
- Logging robusto

Fase 2 - Detección y Tracking:
- Integración RetinaFace
- Implementación ByteTrack
- Gestión tracks

Fase 3 - Reconocimiento:
- Integración ArcFace
- Base datos embeddings
- Multi-frame voting

Fase 4 - Optimización:
- TensorRT inference
- Batch processing
- Memory pooling

## Design Patterns

Aplicados:
- Strategy: Diferentes backends (ONNX/TensorRT)
- Observer: Stats callbacks
- Factory: Config loading
- Singleton: Logger instance

Principios SOLID:
- Single Responsibility
- Open/Closed
- Liskov Substitution
- Interface Segregation
- Dependency Inversion

## Testing

Unit tests por módulo:
- test_stream_capture
- test_draw_utils
- test_utils
- test_config

Integration tests end-to-end:
- test_full_pipeline
- test_recording
- test_viewing

## Debugging

Logs:

```bash
export SPDLOG_LEVEL=debug
./build/bin/panto
```

Profiling:

```bash
perf record -g ./build/bin/panto
perf report
```

Memory:

```bash
valgrind --leak-check=full ./build/bin/panto
```

Visual debugging en config:

```toml
[output]
draw_detections = true
draw_tracks = true
draw_roi = true
draw_fps = true
```