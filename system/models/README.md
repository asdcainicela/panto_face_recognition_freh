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






----------------- update 

#!/bin/bash
# ============= download_fairface_age_gender.sh =============
# Descargar y convertir modelo FairFace Age+Gender
# Compatible con TensorRT en Jetson Orin

set -e

echo "🚀 Instalando dependencias..."
pip3 install -q torch torchvision onnx onnxruntime numpy pillow

echo ""
echo "📥 Descargando modelo FairFace..."

# Crear directorio
mkdir -p /tmp/fairface_model
cd /tmp/fairface_model

# Descargar script de conversión
cat > export_fairface.py << 'PYEOF'
"""
Export FairFace ResNet-34 to ONNX format
Age: 9 classes (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
Gender: 2 classes (Male, Female)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import numpy as np
from collections import OrderedDict

class FairFaceModel(nn.Module):
    def __init__(self, num_age_classes=9, num_gender_classes=2):
        super(FairFaceModel, self).__init__()
        
        # ResNet-34 backbone
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Age head
        self.age_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_age_classes)
        )
        
        # Gender head
        self.gender_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_gender_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        age_logits = self.age_fc(features)
        gender_logits = self.gender_fc(features)
        
        # Concatenar [gender, age] -> shape: [batch, 11]
        # Index 0-1: gender (Male, Female)
        # Index 2-10: age brackets
        output = torch.cat([gender_logits, age_logits], dim=1)
        return output

print("🔨 Creando modelo FairFace...")
model = FairFaceModel()
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

print("📦 Exportando a ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "fairface_age_gender.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "logits": {0: "batch"}
    },
    opset_version=14,
    do_constant_folding=True
)

print("✅ Modelo exportado: fairface_age_gender.onnx")

# Verificar
model_onnx = onnx.load("fairface_age_gender.onnx")
onnx.checker.check_model(model_onnx)
print("✅ Modelo ONNX válido")

# Mostrar info
print("\n📊 Información del modelo:")
print(f"  Input: {model_onnx.graph.input[0].name}")
print(f"  Output: {model_onnx.graph.output[0].name}")
print(f"  Output shape: [batch, 11] (2 gender + 9 age)")
PYEOF

# Ejecutar exportación
python3 export_fairface.py

# ==================== FIJAR BATCH SIZE ====================
echo ""
echo "🔧 Fijando batch size a 1..."

python3 << 'PYEOF'
import onnx

model = onnx.load("fairface_age_gender.onnx")

# Fijar batch=1
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1

onnx.save(model, "fairface_age_gender_fixed.onnx")

print("✅ Batch size fijado")
print("\n📊 Modelo final:")
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  Input: {inp.name} -> {shape}")
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name} -> {shape}")
PYEOF

# ==================== CONVERTIR A TENSORRT ====================
echo ""
echo "🔥 Convirtiendo a TensorRT..."

/usr/src/tensorrt/bin/trtexec \
  --onnx=fairface_age_gender_fixed.onnx \
  --saveEngine=age_gender_fairface.engine \
  --fp16 \
  --workspace=2048 \
  --verbose

if [ $? -ne 0 ]; then
    echo "❌ Error al convertir a TensorRT"
    exit 1
fi

echo "✅ Conversión exitosa"

# ==================== VALIDAR ENGINE ====================
echo ""
echo "🔍 Validando engine..."

/usr/src/tensorrt/bin/trtexec \
  --loadEngine=age_gender_fairface.engine \
  --dumpProfile \
  --verbose

# ==================== COPIAR A MODELS ====================
echo ""
echo "📦 Instalando modelo..."

MODEL_DIR="/workspace/panto_face_recognition_freh/system/models"
mkdir -p "$MODEL_DIR"

# Backup
if [ -f "$MODEL_DIR/age_gender.engine" ]; then
    mv "$MODEL_DIR/age_gender.engine" "$MODEL_DIR/age_gender.engine.old"
fi

cp age_gender_fairface.engine "$MODEL_DIR/age_gender.engine"
cp fairface_age_gender_fixed.onnx "$MODEL_DIR/age_gender.onnx"

echo "✅ Modelo instalado en $MODEL_DIR"

# ==================== LIMPIAR ====================
cd /workspace/panto_face_recognition_freh/system
rm -rf /tmp/fairface_model

echo ""
echo "=" | awk '{for(i=1;i<=80;i++)printf "="}END{print ""}'
echo "✅ ¡INSTALACIÓN COMPLETA!"
echo "=" | awk '{for(i=1;i<=80;i++)printf "="}END{print ""}'
echo ""
echo "📊 Formato del modelo:"
echo "  Input: [1, 3, 224, 224] (RGB, ImageNet normalized)"
echo "  Output: [1, 11]"
echo "    - Index 0-1: Gender (Male, Female)"
echo "    - Index 2-10: Age brackets:"
echo "        [2]: 0-2 years"
echo "        [3]: 3-9 years"
echo "        [4]: 10-19 years"
echo "        [5]: 20-29 years"
echo "        [6]: 30-39 years"
echo "        [7]: 40-49 years"
echo "        [8]: 50-59 years"
echo "        [9]: 60-69 years"
echo "        [10]: 70+ years"
echo ""
echo "🔧 Siguiente paso: Actualizar postprocess() en C++"
echo ""