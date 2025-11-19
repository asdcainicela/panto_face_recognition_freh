import cv2
import numpy as np
import onnxruntime as ort

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

# Cargar imagen
img = cv2.imread("img/frame_0009.jpg")
h, w = img.shape[:2]
print(f"Imagen original: {img.shape}")

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
all_detections = []

# Procesar cada stride
for idx, stride in enumerate(feat_stride_fpn):
    scores = outputs[idx]
    bbox_preds = outputs[idx + fmc] * stride
    kps_preds = outputs[idx + fmc * 2] * stride
    
    height = input_size[1] // stride
    width = input_size[0] // stride
    
    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    
    if num_anchors > 1:
        anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
    
    pos_inds = np.where(scores >= 0.5)[0]
    
    if len(pos_inds) > 0:
        pos_scores = scores[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_kps_preds = kps_preds[pos_inds]
        pos_anchor_centers = anchor_centers[pos_inds]
        
        bboxes = distance2bbox(pos_anchor_centers, pos_bbox_preds)
        kpss = distance2kps(pos_anchor_centers, pos_kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        
        for i in range(len(pos_inds)):
            all_detections.append({
                'bbox': bboxes[i],
                'score': pos_scores[i][0],
                'kps': kpss[i],
                'stride': stride
            })

print(f"\n=== Detecciones encontradas: {len(all_detections)} ===")

# Ordenar por score
all_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)

# Escalar y dibujar
scale_x = w / input_size[0]
scale_y = h / input_size[1]

for i, det in enumerate(all_detections):
    bbox = det['bbox']
    score = det['score']
    kps = det['kps']
    
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    print(f"Cara {i+1}: score={score:.3f}, bbox=[{x1},{y1},{x2},{y2}], tamaño={x2-x1}x{y2-y1}")
    
    # Dibujar bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(img, f"Face {i+1}: {score:.2f}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Dibujar keypoints (5 puntos faciales)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    kp_names = ["Ojo izq", "Ojo der", "Nariz", "Boca izq", "Boca der"]
    
    for j, (kx, ky) in enumerate(kps):
        kx = int(kx * scale_x)
        ky = int(ky * scale_y)
        cv2.circle(img, (kx, ky), 5, colors[j], -1)

print(f"\n✓ Mostrando imagen con detecciones...")
print("Presiona cualquier tecla para cerrar la ventana")

# Mostrar imagen
cv2.imshow('Face Detection - SCRFD', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
