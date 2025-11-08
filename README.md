# panto_face_recognition_freh

Proyecto: Reconocimiento facial en cámaras de vigilancia con distancia media/larga
Autor: AsdCain
Lenguaje: C++17
Hardware objetivo: NVIDIA Jetson Orin Nano
Dependencias principales: OpenCV, TensorRT, ONNXRuntime, InsightFace, GFPGAN o Real-ESRGAN, DeepSort

Descripción general:
----------------------------------------------------
El sistema procesa un stream de video (RTSP, USB o archivo), detecta rostros en tiempo real,
mejora su resolución cuando son pequeños o distantes, genera embeddings faciales de alta precisión,
y realiza comparación con una base de datos local de rostros registrados. Todo el flujo opera sin UI.

Arquitectura de procesamiento:
----------------------------------------------------
1. Captura:
   - Fuente: RTSP o cámara IP.
   - Librería: OpenCV (cv::VideoCapture).
   - Pipeline GStreamer recomendado en Jetson.

2. Detección facial:
   - Modelo: RetinaFace (mobilenet0.25 o r50) exportado a ONNX.
   - Backend: TensorRT (DNN_BACKEND_CUDA / DNN_TARGET_CUDA_FP16).
   - Salida: bounding boxes + keypoints (ojos, nariz, boca).

3. Tracking:
   - Algoritmo: DeepSort (reidentificación + Kalman + IOU).
   - Objetivo: mantener la identidad temporal de cada rostro entre frames.
   - Mejora la estabilidad y reduce llamadas de reconocimiento repetidas.

4. Superresolución facial:
   - Modelos:
     a. GFPGAN (para restauración facial realista).
     b. Real-ESRGAN (para aumento de detalle general).
   - Activación condicional:
     Si el tamaño del rostro < 80x80 px → aplicar SR antes del embedding.
   - Backend: ONNXRuntime o TensorRT (si se convierte el modelo).

5. Extracción de embeddings:
   - Modelo: ArcFace (r100 o r50) en formato ONNX.
   - Entrada: rostro alineado 112x112.
   - Salida: vector float[512] (embedding normalizado).
   - Backend: TensorRT.

6. Comparación facial:
   - Distancia: coseno (1 - dot(A,B)).
   - Umbral típico: < 0.4 = misma persona.
   - Base de datos: SQLite o binario local (JSON/CSV).

7. Almacenamiento y logging:
   - Cada reconocimiento guarda: timestamp, ID de tracking, similitud, nombre.
   - Logs en formato CSV o JSON.
   - Capturas de rostros recortados opcionales.

8. Optimización:
   - Pipeline asincrónico con hilos:
       - Thread 1: captura + detección
       - Thread 2: tracking + SR + embedding
       - Thread 3: comparación + logging
   - Usa memoria unificada (Jetson) y TensorRT FP16 para velocidad.
   - FPS esperado: 25–30 FPS en Orin Nano con 720p.

9. Estructura del proyecto:
   /src
      main.cpp
      detector.cpp / detector.hpp
      tracker.cpp / tracker.hpp
      sr_module.cpp / sr_module.hpp
      recognizer.cpp / recognizer.hpp
      database.cpp / database.hpp
   /models
      retinaface_mnet0.25.onnx
      arcface_r100.onnx
      gfpgan.onnx
   /data
      faces.db
      embeddings/
   /build (CMake)

10. Futuras extensiones:
   - Antispoofing (live face detection).
   - Integración MQTT o REST API local.
   - Fusión con YOLO/DeepSort para conteo de personas.

Resumen:
----------------------------------------------------
El sistema está diseñado para entornos reales de CCTV, donde los rostros son pequeños,
la luz varía y las cámaras están fijas. Combinando superresolución + tracking + ArcFace
se obtiene robustez incluso con rostros parcialmente borrosos o lejanos, sin interfaz gráfica
y con ejecución optimizada para GPU.

