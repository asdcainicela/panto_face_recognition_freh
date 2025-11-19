
import cv2
import numpy as np
import onnxruntime as ort

# Extraer frame del video
print("Extrayendo frame del video...")
cap = cv2.VideoCapture("video1.mp4")
ret, img = cap.read()
cap.release()

if not ret:
    print("❌ Error leyendo video")
    exit(1)

print(f"✓ Frame extraído: {img.shape}")

# Guardar frame para referencia
cv2.imwrite("debug_frame.jpg", img)

def distance2bbox(points, distance, max_shape=None):
    """Decodificar bbox desde distance prediction"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decodificar keypoints desde distance prediction"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

# Cargar modelo
sess = ort.InferenceSession("models/scrfd_10g_bnkps.onnx", 
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

h, w = img.shape[:2]
print(f"Imagen: {w}x{h}")

# Preprocesar
input_size = (640, 640)
img_resized = cv2.resize(img, input_size)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
input_blob = (img_rgb.astype(np.float32) - 127.5) / 128.0
input_blob = np.transpose(input_blob, (2, 0, 1))
input_blob = np.expand_dims(input_blob, axis=0)

# Inferencia
outputs = sess.run(None, {'input.1': input_blob})

# Configuración
fmc = 3
feat_stride_fpn = [8, 16, 32]
num_anchors = 2

print("\n" + "="*80)
print("ANÁLISIS DE OUTPUTS POR STRIDE")
print("="*80)

for idx, stride in enumerate(feat_stride_fpn):
    scores = outputs[idx]
    bbox_preds = outputs[idx + fmc] * stride
    kps_preds = outputs[idx + fmc * 2] * stride
    
    print(f"\n{'='*80}")
    print(f"STRIDE {stride}")
    print(f"{'='*80}")
    print(f"Shape scores: {scores.shape}")
    print(f"Shape bboxes: {bbox_preds.shape}")
    
    # Análisis de scores
    max_score = scores.max()
    num_05 = (scores > 0.5).sum()
    num_07 = (scores > 0.7).sum()
    num_09 = (scores > 0.9).sum()
    
    print(f"\nScores:")
    print(f"  Max score: {max_score:.4f}")
    print(f"  Scores > 0.5: {num_05}")
    print(f"  Scores > 0.7: {num_07}")
    print(f"  Scores > 0.9: {num_09}")
    
    if num_05 > 0:
        # Top 10 scores
        top_indices = np.argsort(scores.flatten())[-10:][::-1]
        top_scores = scores.flatten()[top_indices]
        
        print(f"\n  Top 10 scores:")
        for i, (tidx, sc) in enumerate(zip(top_indices, top_scores)):
            print(f"    {i+1}. Index {tidx}: {sc:.4f}")
        
        # Ver bboxes de los top scores
        height = input_size[1] // stride
        width = input_size[0] // stride
        
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
        
        print(f"\n  Total anchors: {len(anchor_centers)}")
        print(f"  Anchors shape debería ser: {height * width * num_anchors}")
        
        print(f"\n  Análisis de TOP 5 detecciones:")
        for i, tidx in enumerate(top_indices[:5]):
            sc = scores.flatten()[tidx]
            bbox = bbox_preds.reshape(-1, 4)[tidx]
            anchor = anchor_centers[tidx]
            
            # Decodificar bbox
            x1 = anchor[0] - bbox[0]
            y1 = anchor[1] - bbox[1]
            x2 = anchor[0] + bbox[2]
            y2 = anchor[1] + bbox[3]
            
            # Escalar
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
            
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            
            width_box = x2_orig - x1_orig
            height_box = y2_orig - y1_orig
            
            print(f"\n    Detección {i+1} (score={sc:.4f}):")
            print(f"      Anchor center: ({anchor[0]:.1f}, {anchor[1]:.1f})")
            print(f"      BBox distances: L={bbox[0]:.1f}, T={bbox[1]:.1f}, R={bbox[2]:.1f}, B={bbox[3]:.1f}")
            print(f"      BBox (640x640): [{x1:.1f}, {y1:.1f}] -> [{x2:.1f}, {y2:.1f}]")
            print(f"      BBox (original {w}x{h}): [{x1_orig:.0f}, {y1_orig:.0f}] -> [{x2_orig:.0f}, {y2_orig:.0f}]")
            print(f"      Size: {width_box:.0f}x{height_box:.0f}")
            
            # Validaciones
            valid = True
            reasons = []
            
            if width_box < 20 or height_box < 20:
                valid = False
                reasons.append(f"Muy pequeño ({width_box:.0f}x{height_box:.0f})")
            
            if width_box > w * 0.9 or height_box > h * 0.9:
                valid = False
                reasons.append(f"Muy grande ({width_box:.0f}x{height_box:.0f} vs {w}x{h})")
            
            if x1_orig < -50 or y1_orig < -50 or x2_orig > w+50 or y2_orig > h+50:
                valid = False
                reasons.append(f"Muy fuera de límites")
            
            aspect = width_box / height_box if height_box > 0 else 0
            if aspect < 0.4 or aspect > 2.5:
                valid = False
                reasons.append(f"Aspect ratio malo: {aspect:.2f}")
            
            if valid:
                print(f"      ✅ VÁLIDA (aspect={aspect:.2f})")
            else:
                print(f"      ❌ INVÁLIDA: {', '.join(reasons)}")

print("\n" + "="*80)
print("DIAGNÓSTICO FINAL")
print("="*80)

# Contar válidas e inválidas
total_over_05 = sum((outputs[i] > 0.5).sum() for i in range(3))
total_over_07 = sum((outputs[i] > 0.7).sum() for i in range(3))
total_over_09 = sum((outputs[i] > 0.9).sum() for i in range(3))

print(f"Total detecciones > 0.5: {total_over_05}")
print(f"Total detecciones > 0.7: {total_over_07}")
print(f"Total detecciones > 0.9: {total_over_09}")

print("\n¿Qué está pasando?")
if total_over_05 > 50:
    print("❌ PROBLEMA: Demasiadas detecciones!")
    print("\nPosibles causas:")
    print("1. Los anchors se están generando MAL")
    print("   - Deberían ser: feat_h * feat_w * num_anchors")
    print("   - Stride 8:  80*80*2 = 12800")
    print("   - Stride 16: 40*40*2 = 3200")
    print("   - Stride 32: 20*20*2 = 800")
    print(f"\n2. Verificar shapes reales:")
    for idx, stride in enumerate(feat_stride_fpn):
        expected = (640//stride) * (640//stride) * num_anchors
        actual = outputs[idx].shape[0]
        match = "✓" if expected == actual else "✗"
        print(f"   Stride {stride}: esperado={expected}, actual={actual} {match}")
elif total_over_05 < 5:
    print("✓ Cantidad razonable de detecciones")
    print("   Proceder con implementación en C++")
