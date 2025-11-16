# Modelos ONNX

## Descarga

**IMPORTANTE:** Ejecutar desde la raíz del proyecto:

# Para Ubuntu/Debian
apt-get update && apt-get install -y unzip

```bash
cd models
chmod +x install_models.sh
./install_models.sh
cd ..
```

## Modelos Incluidos

| Modelo | Tamaño | Función |
|--------|--------|---------|
| retinaface.onnx | 27 MB | Detección rostros |
| arcface_r100.onnx | 250 MB | Reconocimiento facial |
| realesr_x4.onnx | 67 MB | Super-resolución |

## Conversión TensorRT (Opcional)

```bash
cd models
trtexec --onnx=retinaface.onnx --saveEngine=retinaface.trt --fp16
trtexec --onnx=arcface_r100.onnx --saveEngine=arcface_r100.trt --fp16
trtexec --onnx=realesr_x4.onnx --saveEngine=realesr_x4.trt --fp16
```

TensorRT da 2-3x speedup en Jetson.

convertir onnx a rt primero verificamos el sh, pero tambien el path 

si quieres usar terminal

/usr/src/tensorrt/bin/trtexec --onnx=retinaface.onnx --saveEngine=retinaface.engine --fp16 --workspace=3072

si usamos el archivo c++
primero compilamos con convert_ro_tensorrt.sh
luego ejemplo
./convert_onnx_auto retinaface.onnx retinaface_fp16.engine

