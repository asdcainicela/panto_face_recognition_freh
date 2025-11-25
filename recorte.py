import cv2
import subprocess
import os

# ============= CONFIGURACIÓN =============
start = 0  # Segundo de inicio
end = 84    # Segundo de fin
video_ruta = "output_20251125_152711.mp4"
# =========================================

temp_video = "temp_video.avi"
nombre_salida = f"edit_{video_ruta}"

# Paso 1: Extraer frames con OpenCV
cap = cv2.VideoCapture(video_ruta)
fps = cap.get(cv2.CAP_PROP_FPS)
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_inicio = int(start * fps)
frame_fin = int(end * fps)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video, fourcc, fps, (ancho, alto))

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
print(f"Extrayendo frames de {start}s a {end}s...")

frame_actual = frame_inicio
while frame_actual < frame_fin:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_actual += 1
    
    if frame_actual % 30 == 0:
        progreso = ((frame_actual - frame_inicio) / (frame_fin - frame_inicio)) * 100
        print(f"Progreso: {progreso:.1f}%", end='\r')

cap.release()
out.release()

# Paso 2: Convertir a formato compatible con WhatsApp usando FFmpeg
print("\nConvirtiendo a formato compatible con WhatsApp...")
comando = [
    'ffmpeg', '-i', temp_video,
    '-c:v', 'libx264',
    '-preset', 'medium',
    '-crf', '23',
    '-y', nombre_salida
]

subprocess.run(comando, capture_output=True)
os.remove(temp_video)

print(f"✓ Video guardado como: {nombre_salida}")