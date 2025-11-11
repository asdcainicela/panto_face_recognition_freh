# 🚀 PANTO - Reconocimiento Facial para Control de Acceso

## 📋 DECISIÓN FINAL - NO PENSAR, SOLO HACER

---

## ✅ TU CONFIGURACIÓN ELEGIDA

### **Escenario:**
- 📹 Cámara en puerta de entrada con zoom
- 🎯 Personas pasan por punto fijo (3-5 metros de distancia)
- 💻 Resolución: **1080p (1920x1080)**
- 🖼️ ROI: **50% central** (zona de puerta)
- ⚡ Target: **22-25 FPS**
- 🎲 Precisión esperada: **94-97%**

---

## 🗂️ ARCHIVOS DE CONFIGURACIÓN (5 TOML)

```bash
configs/
├── config_4k.toml           # 3840x2160 - Máxima calidad
├── config_1440p.toml        # 2560x1440 - Alta calidad
├── config_1080p_roi.toml    # 1920x1080 + ROI ⭐ RECOMENDADO
├── config_1080p_full.toml   # 1920x1080 sin ROI
└── config_720p.toml         # 1280x720 - Hardware modesto
```

---

## 📊 TABLA COMPARATIVA COMPLETA

| Resolución | Archivo | GPU Mínima | FPS | Precisión | ROI | SR | Zonas |
|------------|---------|------------|-----|-----------|-----|----|----|
| **4K (3840x2160)** | `config_4k.toml` | RTX 3060 / 2080 | 18-22 | 97-99% | Opcional | Raro | 4 zonas (90-150-220-350px) |
| **1440p (2560x1440)** | `config_1440p.toml` | RTX 2060 / 1660Ti | 20-24 | 96-98% | Opcional | Ocasional | 4 zonas (60-110-160-240px) |
| **1080p + ROI** ⭐ | `config_1080p_roi.toml` | GTX 1650 / RTX 2060M | 22-25 | 94-97% | **Sí** | Condicional | 3 zonas (60-100-200px) |
| **1080p Full** | `config_1080p_full.toml` | GTX 1650 / RTX 2060M | 15-18 | 90-94% | No | Frecuente | 4 zonas (45-80-120-180px) |
| **720p** | `config_720p.toml` | GTX 1050 / MX450 | 18-22 | 86-90% | **Sí** | Casi siempre | 3 zonas (40-70-130px) |

---

## 🎯 DETALLES POR RESOLUCIÓN

### **4K (3840x2160)** - Máxima calidad
```
Rostros:
├─ Más cerca: ~268px (1 dedo = 384px cabeza)
└─ Más lejos: ~90px (1/3 dedo = 128px cabeza)

Zonas:
├─ Far:      90-150px  (track only)
├─ Approach: 150-220px (acumular)
├─ Optimal:  220-350px (DECIDIR) threshold 0.30 ⭐
└─ Close:    >350px    (verificar) threshold 0.25

Super-Resolution: Casi nunca (threshold 150px)
Unknown threshold: 0.55 (estricto)
```

### **1440p (2560x1440)** - Alta calidad
```
Rostros:
├─ Más cerca: ~179px (1 dedo = 256px cabeza)
└─ Más lejos: ~60px (1/3 dedo = 85px cabeza)

Zonas:
├─ Far:      60-110px  (track + SR)
├─ Approach: 110-160px (acumular)
├─ Optimal:  160-240px (DECIDIR) threshold 0.32 ⭐
└─ Close:    >240px    (verificar) threshold 0.28

Super-Resolution: Ocasional (threshold 110px)
Unknown threshold: 0.57
```

### **1080p + ROI** ⭐ - TU CONFIGURACIÓN RECOMENDADA
```
Rostros (con ROI 50% = zoom 2x efectivo):
├─ Más cerca: ~134px efectivo
└─ Más lejos: ~44px efectivo (pero ROI mejora a ~88px)

Zonas:
├─ Far:     60-100px  (track + SR)
├─ Optimal: 100-200px (DECIDIR) threshold 0.35 ⭐
└─ Close:   >200px    (verificar) threshold 0.30

Super-Resolution: Condicional (threshold 80px)
Unknown threshold: 0.60
ROI: 960x540 (50% centrado en puerta)
```

### **1080p Full** - Sin ROI
```
Rostros (sin ROI):
├─ Más cerca: ~134px
└─ Más lejos: ~45px

Zonas:
├─ Far:     45-80px   (track + SR)
├─ Medium:  80-120px  (acumular + SR)
├─ Optimal: 120-180px (DECIDIR) threshold 0.40 ⭐
└─ Close:   >180px    (verificar) threshold 0.35

Super-Resolution: Frecuente (threshold 100px, modelo x4)
Unknown threshold: 0.65
```

### **720p** - Hardware modesto
```
Rostros (con ROI 50%):
├─ Más cerca: ~90px efectivo
└─ Más lejos: ~30px efectivo (ROI mejora a ~60px)

Zonas:
├─ Far:     40-70px  (track + SR)
├─ Optimal: 70-130px (DECIDIR) threshold 0.45 ⭐
└─ Close:   >130px   (verificar) threshold 0.40

Super-Resolution: Casi siempre (threshold 70px, modelo x4)
Unknown threshold: 0.68 (más permisivo)
ROI: 640x360 (50% centrado)
```

---

## ✅ GUÍA DE SELECCIÓN RÁPIDA

### **Tengo GPU potente (RTX 3060+):**
→ Usa `config_4k.toml` si tu cámara soporta 4K  
→ Usa `config_1440p.toml` si tu cámara es 1440p  
→ Máxima precisión (97-99%)

### **Tengo GPU media (GTX 1650, RTX 2060M):** ✅
→ Usa `config_1080p_roi.toml` ⭐ **RECOMENDADO**  
→ Mejor balance velocidad/calidad  
→ 22-25 FPS, precisión 94-97%

### **Tengo GPU básica (GTX 1050, MX450):**
→ Usa `config_720p.toml`  
→ Asegúrate de activar ROI  
→ 18-22 FPS, precisión 86-90%

### **No sé qué GPU tengo:**
→ Empieza con `config_1080p_roi.toml`  
→ Si FPS < 15, baja a `config_720p.toml`  
→ Si FPS > 25, sube a `config_1440p.toml`

**TU ELECCIÓN: `config_1080p_roi.toml`** ✅

---

## 🎯 POR QUÉ ESTA CONFIGURACIÓN

### **1. ROI = Zoom Digital Gratis**
```
Sin ROI:  Procesarías 1920x1080 completo
          Rostros: 45-134 píxeles

Con ROI:  Procesas solo 960x540 (zona de puerta)
          Rostros: 80-220 píxeles (efectivamente "zoom 2x")
          
Resultado: Rostros más grandes SIN cambiar hardware
```

### **2. Super-Resolution Solo Cuando Es Necesario**
```
Persona lejos (rostro 60-100px):
  └─> Aplicar SR ✅
  └─> Tracking + embeddings preliminares

Persona cerca (rostro 100-200px):
  └─> Sin SR ❌ (no es necesario)
  └─> Embedding definitivo
  └─> DECIDIR MATCH ✅
```

### **3. Multi-Frame Voting**
```
No decides en 1 frame, acumulas 10-15 frames:

Frame 10: "Juan" (confianza 48%)
Frame 20: "Juan" (confianza 43%)
Frame 30: "Juan" (confianza 38%)
Frame 40: "Juan" (confianza 32%) ⭐
Frame 50: "Juan" (confianza 28%) ⭐

Votación ponderada → "Juan Pérez, 95% confianza" ✅
```

---

## 🏗️ ARQUITECTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────┐
│                    FRAME 1080p (25 FPS)                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
              ┌──────────────────────┐
              │  APLICAR ROI (50%)   │ ← Procesar solo zona de puerta
              │  960x540 efectivo    │
              └──────────┬───────────┘
                         │
                         v
              ┌──────────────────────┐
              │  DETECTAR ROSTROS    │ ← RetinaFace
              │  Min: 60px           │
              └──────────┬───────────┘
                         │
                         v
              ┌──────────────────────┐
              │  TRACKING (ByteTrack)│ ← Seguir personas
              │  Max 5 simultáneas   │
              └──────────┬───────────┘
                         │
                         v
          ¿Rostro < 80px? ──YES──> ┌──────────────────┐
                 │                  │  SUPER-RESOLUTION│
                 NO                 │  (x2 upscale)    │
                 │                  └────────┬─────────┘
                 │                           │
                 └───────────┬───────────────┘
                             v
                  ┌────────────────────┐
                  │  EXTRAER EMBEDDING │ ← ArcFace R100
                  │  (512 dimensiones) │
                  └─────────┬──────────┘
                            │
                            v
               ┌────────────────────────┐
               │  ACUMULAR EN TRACK     │
               │  (min 10 frames)       │
               └─────────┬──────────────┘
                         │
          ¿En zona óptima? ──YES──> ┌─────────────────┐
          (100-200px)                │  VOTAR Y DECIDIR│
                 │                   │  Match con DB   │
                 NO                  └────────┬────────┘
                 │                            │
                 └────> Seguir tracking       v
                                    ┌──────────────────┐
                                    │  ACCIÓN          │
                                    │  - Log evento    │
                                    │  - Guardar frame │
                                    │  - Webhook/MQTT  │
                                    │  - Abrir puerta  │
                                    └──────────────────┘
```

---

## ⚙️ CONFIGURACIÓN CRÍTICA (Ya está en el TOML)

### **Cámara + ROI:**
```toml
[camera]
resolution = [1920, 1080]
fps_target = 25

[camera.roi]
enabled = true
x = 480      # 25% desde izquierda
y = 270      # 25% desde arriba
width = 960  # 50% del ancho
height = 540 # 50% del alto
```

### **Super-Resolution Condicional:**
```toml
[superresolution]
enabled = true
conditional_threshold = 80  # Solo si rostro < 80px
model = "realesr_x2"        # x2 (no x4, más rápido)
```

### **Zonas de Procesamiento:**
```toml
[zones.far]          # 60-100px: Track + SR
[zones.optimal]      # 100-200px: DECIDIR aquí ⭐
[zones.close]        # >200px: Verificar
```

### **Multi-Frame Voting:**
```toml
[recognition]
strategy = "multi_frame_voting"
min_frames = 10              # Esperar 10 frames
confidence_threshold = 0.35  # Threshold en zona óptima
```

---

## 🚀 CÓMO EJECUTAR

### **1. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

### **2. Descargar modelos:**
```bash
python scripts/download_models.py
```

Modelos necesarios:
- `retinaface_mobilenet.onnx` (detección)
- `arcface_r100.onnx` (embeddings)
- `realesr_x2.onnx` (super-resolution)

### **3. Preparar base de datos:**
```bash
python scripts/setup_database.py
python scripts/add_person.py --name "Juan Pérez" --photos ./photos/juan/
```

### **4. Ajustar ROI (si es necesario):**
```bash
# Herramienta para visualizar y ajustar ROI
python scripts/calibrate_roi.py --config configs/config_1080p_roi.toml
```

### **5. Ejecutar:**
```bash
python main.py --config configs/config_1080p_roi.toml
```

---

## 📊 MÉTRICAS ESPERADAS

### **Performance:**
```
FPS: 22-25 constante ✅
Latencia detección: 10-20ms
Latencia embedding: 15-25ms
Latencia SR (cuando aplica): 40-60ms
Latencia total por frame: 50-80ms

Decisión de match: 1-2 segundos (acumular 10-15 frames)
```

### **Precisión:**
```
Zona óptima (100-200px): 95-97% ✅
Zona lejana (con SR):    90-92%
Promedio general:        94-96%

Falsos positivos: <2%
Falsos negativos: <3%
```

### **Memoria:**
```
GPU VRAM: 2-3 GB
RAM: 2-4 GB
Disk (logs/capturas): ~100MB/día
```

---

## 🎛️ AJUSTES RÁPIDOS

### **Si FPS < 20:**
```toml
# Opción 1: Desactivar SR
[superresolution]
enabled = false

# Opción 2: ROI más pequeño (más zoom)
[camera.roi]
width = 640   # 33% en vez de 50%
height = 360

# Opción 3: Menos threads
[performance]
detector_threads = 1
recognition_threads = 1
```

### **Si muchos falsos positivos:**
```toml
# Ser más estricto
[zones.optimal]
confidence_threshold = 0.30  # Bajar de 0.35 a 0.30

[matching]
unknown_person_threshold = 0.65  # Subir de 0.60 a 0.65
```

### **Si muchos falsos negativos:**
```toml
# Ser más permisivo
[zones.optimal]
confidence_threshold = 0.40  # Subir de 0.35 a 0.40

[recognition]
min_frames = 8  # Bajar de 10 a 8
```

---

## 🔧 CALIBRACIÓN INICIAL

### **Paso 1: Verificar ROI**
```bash
python scripts/calibrate_roi.py --config configs/config_1080p_roi.toml
```
- Asegúrate que el ROI cubra toda la zona de puerta
- Ajusta `x, y, width, height` si es necesario

### **Paso 2: Medir tamaños de rostro**
```bash
python scripts/measure_faces.py --config configs/config_1080p_roi.toml
```
- Párate en diferentes posiciones
- Verifica que rostros estén entre 60-220px
- Si son más pequeños: aumenta ROI zoom

### **Paso 3: Calibrar thresholds**
```bash
python scripts/calibrate_thresholds.py --config configs/config_1080p_roi.toml
```
- Registra 10-20 pasadas de personas conocidas
- Script sugiere thresholds óptimos

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
panto/
├── configs/
│   └── config_1080p_roi.toml      ⭐ TU ARCHIVO
├── models/
│   ├── retinaface_mobilenet.onnx
│   ├── arcface_r100.onnx
│   └── realesr_x2.onnx
├── data/
│   ├── known_faces.db             # Base de datos de personas
│   ├── panto.db                   # Eventos de acceso
│   └── captures/                  # Fotos guardadas
├── logs/
│   ├── panto.log                  # Log principal
│   └── access_events.db           # Eventos SQL
├── scripts/
│   ├── download_models.py
│   ├── setup_database.py
│   ├── add_person.py
│   ├── calibrate_roi.py
│   ├── measure_faces.py
│   └── calibrate_thresholds.py
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── recognizer.py
│   ├── superresolution.py
│   └── matcher.py
├── main.py
└── requirements.txt
```

---

## 🎯 INTEGRACIÓN CON SISTEMA DE ACCESO

### **Webhook (HTTP):**
```toml
[actions.webhook]
enabled = true
url = "http://192.168.1.10:5000/access"
on_match = true

payload_template = '''
{
    "person_id": "{{ person_id }}",
    "person_name": "{{ person_name }}",
    "confidence": {{ confidence }},
    "timestamp": "{{ timestamp }}",
    "access_granted": {{ is_known }}
}
'''
```

### **MQTT (IoT/ESP32):**
```toml
[actions.mqtt]
enabled = true
broker = "192.168.1.10"
port = 1883
topic_prefix = "panto/access"

# Publicará a: panto/access/match o panto/access/unknown
```

### **GPIO (Raspberry Pi - control directo de puerta):**
```python
# En src/actions.py
def on_match(person_id, confidence):
    if confidence > 0.90:
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # Abrir puerta
        time.sleep(3)
        GPIO.output(RELAY_PIN, GPIO.LOW)   # Cerrar
```

---

## 📊 MONITOREO

### **Dashboard web (opcional):**
```bash
python scripts/web_dashboard.py --port 8080
```
- Ver eventos en tiempo real
- Estadísticas de accesos
- Gestionar personas en DB

### **Logs:**
```bash
tail -f logs/panto.log                    # Log principal
sqlite3 logs/access_events.db "SELECT * FROM access_events ORDER BY timestamp DESC LIMIT 10"
```

---

## ✅ CHECKLIST DE DEPLOYMENT

- [ ] GPU drivers instalados (CUDA/cuDNN)
- [ ] Modelos descargados en `models/`
- [ ] Base de datos inicializada
- [ ] Al menos 3 personas registradas (para pruebas)
- [ ] ROI calibrado (visualizado con `calibrate_roi.py`)
- [ ] Thresholds ajustados (con `calibrate_thresholds.py`)
- [ ] Webhook/MQTT configurado (si aplica)
- [ ] Prueba con persona conocida → Match ✅
- [ ] Prueba con persona desconocida → Unknown ✅
- [ ] FPS > 20 constante

---

## 🚨 TROUBLESHOOTING

### **FPS bajo (<15):**
1. Desactivar SR: `superresolution.enabled = false`
2. Reducir ROI: `width=640, height=360`
3. Cambiar modelo detector: `detection.model = "scrfd_500m"`

### **No detecta rostros:**
1. Verificar ROI: `output.draw_roi = true`
2. Bajar threshold: `detection.confidence_threshold = 0.5`
3. Reducir min size: `detection.min_face_size = 40`

### **Muchos falsos positivos:**
1. Subir threshold: `zones.optimal.confidence_threshold = 0.30`
2. Más frames: `recognition.min_frames = 15`
3. Mejor calidad DB: Re-registrar personas con más fotos

### **Muchos falsos negativos:**
1. Bajar threshold: `zones.optimal.confidence_threshold = 0.40`
2. Menos frames: `recognition.min_frames = 8`
3. Activar SR siempre: `superresolution.conditional_threshold = 200`

---

## 📞 SOPORTE

- Documentación: `docs/`
- Issues: GitHub Issues
- Logs: `logs/panto.log`

---

## 🎉 RESUMEN: YA ESTÁ DECIDIDO

### **Lo que tienes:**
✅ Config optimizada para tu caso: `config_1080p_roi.toml`  
✅ ROI = zoom 2x en zona de puerta  
✅ SR condicional (solo cuando es necesario)  
✅ Multi-frame voting (10 frames)  
✅ 22-25 FPS esperados  
✅ 94-97% precisión esperada  

### **Lo que NO tienes que pensar:**
❌ Qué resolución usar → **1080p con ROI**  
❌ Si usar SR o no → **Sí, pero condicional**  
❌ Cuántos frames votar → **10 frames**  
❌ Qué thresholds → **Ya están configurados**  

### **Solo hacer:**
1. Usar `config_1080p_roi.toml`
2. Calibrar ROI (1 vez)
3. Registrar personas
4. Ejecutar

**No pensar más. Solo implementar.** 🚀