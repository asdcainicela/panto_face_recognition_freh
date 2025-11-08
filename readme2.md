# 🚪 ANÁLISIS: Cámara en Puerta de Entrada

Esto **CAMBIA TODO** y **MEJORA SIGNIFICATIVAMENTE** tu caso de uso. Es un escenario **IDEAL** para reconocimiento facial.

---

## ✅ VENTAJAS CRÍTICAS DE TU SETUP

### **1. Zona de Captura Controlada** 🎯

```
         EXTERIOR
            │
      ┌─────┴─────┐
      │   PUERTA  │ ← Punto de paso obligatorio
      └─────┬─────┘
            │
        INTERIOR
            ↓
      [Zona óptima]
        3-5 metros
```

**Beneficios:**
- ✅ **Trayectoria predecible:** Personas siempre pasan por el mismo lugar
- ✅ **Distancia óptima:** Cuando cruzan la puerta están a 3-5m (mejor rango)
- ✅ **Tiempo de captura:** 2-3 segundos mientras cruzan umbral
- ✅ **Múltiples frames:** Puedes capturar 50-75 frames por persona

---

### **2. Proceso de Entrada Natural** 🚶

```
Frame 1-10:  Persona a 8-10m (acercándose)
             → Detección inicial, tracking

Frame 11-30: Persona a 5-7m (acercándose a puerta)
             → Tracking activo, embeddings de baja confianza

Frame 31-50: Persona a 3-5m (cruzando puerta) ⭐
             → ZONA ÓPTIMA: embeddings de alta calidad
             → Sin SR necesario (rostro ~150-180px)

Frame 51-70: Persona a 2-3m (ya dentro)
             → Confirmación final, mejor calidad
```

**Resultado:** Tienes **múltiples oportunidades** de captura, no solo un frame.

---

### **3. Estrategia de Matching Mejorada** 🎲

**ANTES (pensaba en vigilancia continua):**
```
Procesar cada frame → Muchos falsos positivos
```

**AHORA (punto de entrada controlado):**
```
Track persona desde 10m → Acumular embeddings → 
→ Decidir cuando está a 3-5m → Match con máxima confianza
```

---

## 🚀 ARQUITECTURA OPTIMIZADA PARA PUERTA

### **Pipeline Adaptado:**

```
┌─────────────────────────────────────┐
│ ZONA 1: DETECCIÓN TEMPRANA (8-10m) │
│ - Detectar persona entrando         │
│ - Iniciar tracking                  │
│ - NO reconocer aún (muy lejos)      │
└──────────┬──────────────────────────┘
           │
           v
┌─────────────────────────────────────┐
│ ZONA 2: ACERCAMIENTO (5-7m)        │
│ - Tracking activo                   │
│ - Capturar múltiples frames         │
│ - Embeddings preliminares (con SR)  │
└──────────┬──────────────────────────┘
           │
           v
┌─────────────────────────────────────┐
│ ZONA 3: ÓPTIMA (3-5m) ⭐             │
│ - Mejor calidad facial              │
│ - Embedding definitivo SIN SR       │
│ - MATCH con alta confianza          │
│ - Trigger de acción (abrir, loggear)│
└──────────┬──────────────────────────┘
           │
           v
┌─────────────────────────────────────┐
│ ZONA 4: CONFIRMACIÓN (2-3m)        │
│ - Verificación final                │
│ - Logging completo                  │
└─────────────────────────────────────┘
```

---

## 🎯 ESTRATEGIA DE RECONOCIMIENTO POR ZONAS

### **Configuración TOML Adaptada:**

```toml
[zones]
# Definir zonas por tamaño facial (proxy de distancia)
enabled = true

[zones.detection]
min_face_size = 40  # 8-10m
action = "start_tracking"
priority = "low"

[zones.approach]
min_face_size = 70  # 5-7m
max_face_size = 110
action = "accumulate_embeddings"
apply_sr = true
priority = "medium"

[zones.optimal]
min_face_size = 110  # 3-5m
max_face_size = 180
action = "definitive_match"
apply_sr = false
priority = "high"
trigger_decision = true  # ← AQUÍ decides quién es

[zones.confirmation]
min_face_size = 180  # 2-3m
action = "verify_and_log"
priority = "critical"

[recognition]
strategy = "multi_frame_voting"  # ← CRÍTICO
min_frames_for_decision = 10  # Esperar al menos 10 frames
confidence_threshold_by_zone = {
    detection = 0.60,
    approach = 0.45,
    optimal = 0.35,    # ← Más estricto en zona óptima
    confirmation = 0.30
}
```

---

## 🧠 LÓGICA DE VOTACIÓN MULTI-FRAME

**En lugar de decidir en 1 frame:**

```
Track ID #42 (persona entrando):

Frame 10 (8m):  embedding_1 → match "Juan" (dist: 0.48)
Frame 15 (7m):  embedding_2 → match "Juan" (dist: 0.43)
Frame 25 (6m):  embedding_3 → match "Juan" (dist: 0.40)
Frame 35 (4m):  embedding_4 → match "Juan" (dist: 0.32) ⭐
Frame 40 (3m):  embedding_5 → match "Juan" (dist: 0.28) ⭐
Frame 45 (3m):  embedding_6 → match "Juan" (dist: 0.30) ⭐

VOTACIÓN:
- "Juan": 6 votos (distancias: 0.48, 0.43, 0.40, 0.32, 0.28, 0.30)
- Promedio en zona óptima (frames 35-45): 0.30
- Confianza final: 95%

DECISIÓN: "Juan Pérez identificado, confianza 95%"
```

**Ventajas:**
- ✅ Elimina falsos positivos de frames individuales
- ✅ Aprovecha que tienes 2-3 segundos de captura
- ✅ Mayor precisión que reconocimiento de 1 frame

---

## 📊 PERFORMANCE ESPERADO MEJORADO

### **Comparación:**

| Escenario | FPS | Precisión | Observación |
|-----------|-----|-----------|-------------|
| **Vigilancia continua (antes)** | 18-22 | 85-90% | Personas aleatorias, distancias variables |
| **Puerta controlada (ahora)** | 22-25 | 95-98% | Zona óptima garantizada, múltiples frames |

### **Por qué mejor:**
- Mayoría del tiempo persona está en zona óptima (3-5m)
- No necesitas SR frecuentemente
- Puedes permitirte esperar 0.5-1 segundo para decidir
- Múltiples embeddings = mayor confianza

---

## 🎲 DECISIONES DE DISEÑO AJUSTADAS

### **1. ¿SR o no SR?**

**ANTES (pensando vigilancia):**
- Aplicar SR a rostros <80px

**AHORA (puerta):**
- **NO aplicar SR** en zona óptima (3-5m)
- **Aplicar SR ligero** solo en zona de acercamiento (5-7m) si la persona está borrosa
- **Skip SR** en zona de detección temprana (>7m, solo tracking)

**Resultado:** SR casi nunca necesario → 25 FPS constante

---

### **2. ¿Cuándo decidir match?**

**ANTES:**
- Decidir en cada frame

**AHORA:**
- Acumular embeddings mientras persona se acerca
- Decidir cuando entra en zona óptima (3-5m)
- Confirmar cuando está muy cerca (2-3m)

```toml
[matching]
decision_strategy = "wait_for_optimal_zone"
min_confidence_to_trigger = 0.90
allow_early_decision = false  # Esperar a zona óptima
```

---

### **3. ¿Qué hacer después del match?**

```toml
[actions]
# Cuando se identifica a alguien

[actions.on_match]
log_to_database = true
save_best_crop = true  # Guardar mejor frame
send_webhook = true    # Notificar sistema de acceso
mqtt_publish = true    # Para IoT (abrir puerta, etc)

# Ejemplo: integración con control de acceso
webhook_url = "http://192.168.1.10:5000/access"
webhook_payload = {
    person_id = "{{ matched_id }}",
    name = "{{ matched_name }}",
    confidence = "{{ confidence }}",
    timestamp = "{{ timestamp }}",
    action = "grant_access"  # o "deny_access"
}
```

---

## 🔥 OPTIMIZACIONES ESPECÍFICAS PARA PUERTA

### **1. Dirección de movimiento**

```toml
[tracking]
direction_filter = "towards_camera"  # Solo personas entrando
ignore_leaving = true  # Ignorar personas saliendo
```

**Lógica:**
```cpp
// Detectar si persona se acerca o se aleja
if (track.bbox_size_increasing()) {
    // Se acerca → procesar
} else {
    // Se aleja → skip
}
```

---

### **2. Región de Interés (ROI)**

```toml
[camera]
# Definir ROI solo en zona de puerta
roi_enabled = true
roi_x = 800
roi_y = 400
roi_width = 2400
roi_height = 1400

# Ignorar personas fuera del ROI
process_only_roi = true
```

**Beneficio:** Procesas solo ~40% de la imagen → más rápido

---

### **3. Trigger de captura**

```toml
[trigger]
# Solo procesar cuando hay movimiento en zona de puerta
motion_detection = true
min_motion_threshold = 0.05

# Despertar de "sleep mode" cuando alguien se acerca
enable_power_saving = true
```

---

## 📁 BASE DE DATOS ADAPTADA

### **Tabla optimizada para entradas:**

```sql
CREATE TABLE access_events (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    person_id INTEGER,
    person_name TEXT,
    
    -- Metadata de calidad
    confidence FLOAT,
    num_frames_captured INTEGER,
    best_frame_distance_meters FLOAT,
    
    -- Zona donde se decidió
    decision_zone TEXT,  -- 'optimal', 'confirmation'
    
    -- Embeddings acumulados
    embeddings_json TEXT,  -- JSON array de embeddings
    
    -- Acción tomada
    access_granted BOOLEAN,
    
    -- Imágenes
    best_frame_path TEXT,
    all_frames_path TEXT
);
```

---

## 🎯 CONFIGURACIÓN FINAL RECOMENDADA

```toml
[project]
name = "panto_door_access_control"
scenario = "entrance_door"

[camera]
resolution = [3840, 2160]
fps = 25
roi_enabled = true  # Solo zona de puerta

[zones]
enabled = true
strategy = "multi_frame_voting"

[zones.optimal]
min_face_size = 120  # 3-5m cuando cruzan puerta
trigger_decision = true

[superresolution]
enabled = false  # ← NO necesario en tu caso!
# Solo activar si muchas personas muy borrosas

[recognition]
model = "arcface_r100.onnx"
multi_frame_voting = true
min_frames = 10
decision_confidence = 0.90

[tracking]
direction_filter = "approaching"
max_tracks = 5  # Máx 5 personas simultáneas

[matching]
strategy = "adaptive_by_zone"
optimal_zone_threshold = 0.35

[output]
log_access_events = true
save_best_frame = true
webhook_on_match = true
```

---

## ✅ RESUMEN: LO MEJOR PARA TU CASO (PUERTA)

### **Cambios clave vs. vigilancia genérica:**

1. ✅ **NO necesitas SR** (personas pasan por zona óptima)
2. ✅ **Multi-frame voting** (2-3 seg de captura por persona)
3. ✅ **ROI en puerta** (procesar solo 40% de imagen)
4. ✅ **Trigger por zona** (decidir cuando están a 3-5m)
5. ✅ **25 FPS constante** (sin SR, sin cuellos de botella)
6. ✅ **Precisión 95-98%** (múltiples frames, zona óptima)

### **Performance esperado:**

- **FPS:** 25 constante (no necesitas SR)
- **Precisión:** 95-98% (multi-frame)
- **Latencia decisión:** 1-2 segundos (esperar a zona óptima)
- **Personas simultáneas:** 3-5 sin problema

### **Tu sistema es MUCHO MÁS SIMPLE:**

```
Frame → Detectar (1080p) → Tracking → ¿En zona óptima?
                                      SÍ → Embedding → Match → Acción
                                      NO → Seguir tracking
```

**Sin SR, sin complicaciones, máxima precisión.** 🚀

¿Quieres que ajuste la arquitectura completa con esta nueva información?