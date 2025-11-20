# Modelos ONNX

wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx

wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/googlenet/model/age_googlenet.onnx



wget https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt -O models/age_dex.prototxt


https://huggingface.co/onnxmodelzoo/emotion-ferplus-8?utm_source=chatgpt.com
https://huggingface.co/onnx-community/age-gender-prediction-ONNX?utm_source=chatgpt.com

estos dos uno para ver info y otro para descargar
https://github.com/xlite-dev/ssrnet-toolkit?utm_source=chatgpt.com
https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/tree/main


SCRFD te da la detección de rostro, ArcFace te da el reconocimiento, y FER+ te da las emociones.
El modelo age-gender te da la edad estimada y el género.
Con esos cuatro modelos se cubre detección, identidad, emoción, edad y género al 100%.

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


# Ver detalles del modelo FER+
python3 -c "
import onnx
model = onnx.load('emotion-ferplus-8.onnx')
print('FER+ Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
print('FER+ Output:', model.graph.output[0].name, model.graph.output[0].type.tensor_type.shape)
"

# Ver detalles del modelo Age-Gender
python3 -c "
import onnx
model = onnx.load('model_q4.onnx')
print('AgeGender Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
print('AgeGender Output:', [o.name for o in model.graph.output])
"


cd ~/jetson_workspace/panto_face_recognition_freh/system/models

# FER+ (Emotion Recognition) - 64x64 grayscale
/usr/src/tensorrt/bin/trtexec \
  --onnx=emotion-ferplus-8.onnx \
  --saveEngine=emotion_ferplus.engine \
  --fp16 \
  --workspace=2048 \
  --minShapes=Input3:1x1x64x64 \
  --optShapes=Input3:1x1x64x64 \
  --maxShapes=Input3:1x1x64x64

# Age-Gender (usar el FP16 para mejor precisión)
/usr/src/tensorrt/bin/trtexec \
  --onnx=model_fp16.onnx \
  --saveEngine=age_gender.engine \
  --fp16 \
  --workspace=2048 \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:1x3x224x224 \
  --maxShapes=pixel_values:1x3x224x224


  root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# python3 -c "
> import onnx
> model = onnx.load('model_q4.onnx')
> print('AgeGender Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
> print('AgeGender Output:', [o.name for o in model.graph.output])
> "
AgeGender Input: pixel_values dim {
  dim_param: "batch_size"
}
dim {
  dim_param: "num_channels"
}
dim {
  dim_param: "height"
}
dim {
  dim_param: "width"
}

AgeGender Output: ['logits']
root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# python3 -c "
> import onnx
> model = onnx.load('emotion-ferplus-8.onnx')
> print('FER+ Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
> print('FER+ Output:', model.graph.output[0].name, model.graph.output[0].type.tensor_type.shape)
> "
FER+ Input: Input3 dim {
  dim_value: 1
}
dim {
  dim_value: 1
}
dim {
  dim_value: 64
}
dim {
  dim_value: 64
}

FER+ Output: Plus692_Output_0 dim {
  dim_value: 1
}
dim {
  dim_value: 8
}



```markdown
SCRFD te da la detección de rostro, ArcFace te da el reconocimiento, y FER+ te da las emociones.
El modelo age-gender te da la edad estimada y el género.
Con esos cuatro modelos se cubre detección, identidad, emoción, edad y género al 100%.

```

ese codigo ya tiene dos modelos y asi que seria agregar dos modelos mas en el codigo y que sea completo

ojo todo sera en engine
debo usar algo asi para los dos modelos que falta el engine?

```markdown
/usr/src/tensorrt/bin/trtexec --onnx=retinaface.onnx --saveEngine=retinaface.engine --fp16 --workspace=3072
```
root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# python3 -c "
> import onnx
> model = onnx.load('model_q4.onnx')
> print('AgeGender Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
> print('AgeGender Output:', [o.name for o in model.graph.output])
> "
AgeGender Input: pixel_values dim {
  dim_param: "batch_size"
}
dim {
  dim_param: "num_channels"
}
dim {
  dim_param: "height"
}
dim {
  dim_param: "width"
}
AgeGender Output: ['logits']
root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# python3 -c "
> import onnx
> model = onnx.load('emotion-ferplus-8.onnx')
> print('FER+ Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
> print('FER+ Output:', model.graph.output[0].name, model.graph.output[0].type.tensor_type.shape)
> "
FER+ Input: Input3 dim {
  dim_value: 1
}
dim {
  dim_value: 1
}
dim {
  dim_value: 64
}
dim {
  dim_value: 64
}
FER+ Output: Plus692_Output_0 dim {
  dim_value: 1
}
dim {
  dim_value: 8
}
root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models#

es deecir usar ager-gender-prediction -ONNX y tegnpo archivos model_fp16 int8, etc etc


[11/20/2025-08:32:52] [E] Engine set up failed

&&&& FAILED TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --onnx=emotion-ferplus-8.onnx --saveEngine=emotion_ferplus.engine --fp16 --workspace=2048 --minShapes=Input3:1x1x64x64 --optShapes=Input3:1x1x64x64 --maxShapes=Input3:1x1x64x64

root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# 


[11/20/2025-08:35:11] [I] Explanations of the performance metrics are printed in the verbose logs.
[11/20/2025-08:35:11] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --onnx=model_fp16.onnx --saveEngine=age_gender.engine --fp16 --workspace=2048 --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:1x3x224x224 --maxShapes=pixel_values:1x3x224x224
root@jorinbriq06:/workspace/panto_face_recognition_freh/system/models# 
