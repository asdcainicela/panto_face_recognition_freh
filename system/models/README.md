# onnx models installation script

apt-get update && apt-get install -y unzip

```bash
cd models
chmod +x install_models.sh
./install_models.sh
cd ..
```

## instalacion manual de  Modelos ONNX

https://huggingface.co/onnxmodelzoo/emotion-ferplus-8?utm_source=chatgpt.com
https://huggingface.co/onnx-community/age-gender-prediction-ONNX?utm_source=chatgpt.com


## Convertir modelos a TensorRT Engines (FP16)

### ejemplo basico
trtexec --onnx=retinaface.onnx --saveEngine=retinaface.trt --fp16
trtexec --onnx=arcface_r100.onnx --saveEngine=arcface_r100.trt --fp16
trtexec --onnx=realesr_x4.onnx --saveEngine=realesr_x4.trt --fp16

/usr/src/tensorrt/bin/trtexec --onnx=retinaface.onnx --saveEngine=retinaface.engine --fp16 --workspace=3072


/usr/src/tensorrt/bin/trtexec \
  --onnx=emotion-ferplus-8.onnx \
  --saveEngine=emotion_ferplus.engine \
  --fp16 \
  --workspace=2048

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

------

  /usr/src/tensorrt/bin/trtexec \
  --onnx=model_fp16.onnx \
  --saveEngine=age_gender.engine \
  --fp16 \
  --workspace=2048 

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



python3 - << 'EOF'
import onnx
model = onnx.load('model_q4.onnx')
print('model Input:', model.graph.input[0].name, model.graph.input[0].type.tensor_type.shape)
print('AgeGender Output:', [o.name for o in model.graph.output])
EOF
