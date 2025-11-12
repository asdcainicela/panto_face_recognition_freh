# Resumen de Refactorizacion PANTO

## Archivos Creados/Modificados

### Nuevos Headers
1. **include/draw_utils.hpp**
   - Funciones de dibujo separadas de la logica
   - DrawConfig para configurar estilos
   - draw_stream_info(), draw_recording_indicator(), etc.

2. **include/config.hpp**
   - Constantes centralizadas
   - Configuracion de camara, display, recording
   - Deteccion automatica de resolucion

### Nuevos Source Files
3. **src/draw_utils.cpp**
   - Implementacion de funciones de dibujo
   - Sin mezcla con logica de captura

4. **src/utils.cpp (refactored)**
   - Usa Config:: en lugar de magic numbers
   - Codigo mas limpio y mantenible

### Archivos Refactorizados
5. **src/stream_capture.cpp**
   - Removido todo putText/rectangle
   - Usa Config:: para constantes
   - Solo logica de captura/grabacion

6. **test/record.cpp**
   - Usa Config:: para defaults
   - Sin hardcoded strings

7. **test/view.cpp**
   - Usa DrawUtils para display
   - Usa Config:: para defaults

8. **test/record_view.cpp**
   - Usa DrawUtils + Config
   - Codigo mas limpio

9. **src/main.cpp**
   - Usa DrawUtils para visualizacion
   - Usa Config:: para defaults
   - Deteccion automatica de resolucion

10. **CMakeLists.txt**
    - Incluye draw_utils.cpp en libreria
    - Estructura mas organizada

---

## Principios Aplicados

### Separacion de Responsabilidades
- **StreamCapture**: Solo captura/grabacion
- **DrawUtils**: Solo visualizacion/dibujo
- **Config**: Solo configuracion/constantes
- **Tests**: Combinan componentes

### DRY (Don't Repeat Yourself)
- Constantes en Config::
- Funciones de dibujo reutilizables
- No duplicar codigo entre tests

### SOLID (Simplificado)
- Single Responsibility: Cada clase hace una cosa
- Open/Closed: Facil extender sin modificar
- No funciones innecesarias
- Codigo compacto pero legible

---

## Estructura Final

```
panto/
├── include/
│   ├── config.hpp          [NEW] Constantes centralizadas
│   ├── draw_utils.hpp      [NEW] Funciones de dibujo
│   ├── stream_capture.hpp  [CLEAN] Solo captura
│   └── utils.hpp           [CLEAN] Utilidades basicas
├── src/
│   ├── config.cpp          [FUTURE] Parser TOML
│   ├── draw_utils.cpp      [NEW] Implementacion dibujo
│   ├── stream_capture.cpp  [REFACTORED] Sin display logic
│   ├── utils.cpp           [REFACTORED] Usa Config::
│   └── main.cpp            [REFACTORED] Usa DrawUtils
├── test/
│   ├── record.cpp          [REFACTORED] Usa Config
│   ├── view.cpp            [REFACTORED] Usa DrawUtils+Config
│   └── record_view.cpp     [REFACTORED] Usa DrawUtils+Config
└── build/
    ├── bin/
    │   ├── record
    │   ├── view
    │   ├── record_view
    │   └── panto
    └── lib/
        └── libstream_lib.so
```

---

## Ventajas

1. **Mantenibilidad**: Cambiar colores/estilos? Solo editar DrawUtils
2. **Reutilizacion**: DrawUtils se usa en todos los tests
3. **Configuracion**: Cambiar defaults? Solo editar Config
4. **Testing**: Facil testear componentes por separado
5. **Escalabilidad**: Agregar nuevas funciones sin tocar codigo existente

---

## Ejemplo de Uso

### Antes (Hardcoded)
```cpp
cv::putText(frame, "stream: main", cv::Point(10, 20), 
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
// Repetido en cada archivo...
```

### Despues (Refactorizado)
```cpp
DrawUtils::DrawConfig config;
DrawUtils::draw_stream_info(frame, stats, stream_type, config);
```

---

## Siguientes Pasos

1. Parser de TOML en config.cpp
2. Integracion con detector/tracker
3. Mas funciones de dibujo si necesario
4. Tests unitarios

---

## Como Compilar

```bash
./build.sh
```

Genera:
- `libstream_lib.so` con stream_capture + draw_utils + utils + config
- Ejecutables que usan la libreria

---

## Como Ejecutar

```bash
# Script rapido
./run.sh 1          # record main
./run.sh 3          # view both
./run.sh 6 60       # record+view 60s

# Manual
./build/bin/record main 60
./build/bin/view
./build/bin/record_view sub
./build/bin/panto --config configs/config_1080p_roi.toml
```