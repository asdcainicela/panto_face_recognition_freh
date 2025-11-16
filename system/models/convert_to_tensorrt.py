#!/usr/bin/env python3
"""
Convertir modelos ONNX de Buffalo a TensorRT engines
Optimizado para Jetson con 8GB RAM
"""

import tensorrt as trt
import os
import sys
import argparse
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=2, batch_size=1):
    """
    Construir TensorRT engine desde ONNX
    
    Args:
        onnx_path: Ruta al archivo .onnx
        engine_path: Ruta donde guardar el .engine
        fp16: Usar precisión FP16 (más rápido, menos memoria)
        workspace_gb: Memoria de trabajo en GB (2GB para Jetson 8GB)
        batch_size: Tamaño de batch
    """
    print(f"\n{'='*60}")
    print(f"Convirtiendo: {onnx_path}")
    print(f"Destino: {engine_path}")
    print(f"FP16: {fp16}, Workspace: {workspace_gb}GB, Batch: {batch_size}")
    print(f"{'='*60}\n")
    
    # Crear builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parsear ONNX
    print("Parseando ONNX...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR al parsear ONNX:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return False
    
    print(f"✓ ONNX parseado exitosamente")
    print(f"  Inputs: {network.num_inputs}")
    print(f"  Outputs: {network.num_outputs}")
    
    # Mostrar info de inputs/outputs
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name} - Shape: {inp.shape} - Dtype: {inp.dtype}")
    
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} - Shape: {out.shape} - Dtype: {out.dtype}")
    
    # Configurar builder
    config = builder.create_builder_config()
    
    # Memoria de trabajo (ajustar según RAM disponible)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    
    # FP16 para Jetson (mucho más rápido)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 habilitado")
    else:
        print("⚠ FP16 no disponible, usando FP32")
    
    # Optimización para inferencia
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    
    # Construir engine
    print("\nConstruyendo TensorRT engine...")
    print("⏳ Esto puede tomar varios minutos...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("✗ ERROR: No se pudo construir el engine")
        return False
    
    # Guardar engine
    print(f"\n✓ Engine construido exitosamente")
    print(f"Guardando en: {engine_path}")
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✓ Engine guardado ({file_size_mb:.2f} MB)")
    
    return True


def convert_buffalo_models(models_dir, output_dir, fp16=True, workspace_gb=2):
    """
    Convertir todos los modelos Buffalo ONNX a TensorRT
    
    Buffalo tiene típicamente:
    - det_10g.onnx (detector de caras)
    - w600k_r50.onnx (recognition embedding)
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar archivos ONNX
    onnx_files = list(models_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"✗ No se encontraron archivos .onnx en {models_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Encontrados {len(onnx_files)} modelos ONNX:")
    for f in onnx_files:
        print(f"  - {f.name}")
    print(f"{'='*60}\n")
    
    success_count = 0
    
    for onnx_file in onnx_files:
        engine_file = output_dir / f"{onnx_file.stem}.engine"
        
        # Ajustar parámetros según el modelo
        model_workspace = workspace_gb
        
        # El detector suele ser más grande, darle más memoria si es necesario
        if 'det' in onnx_file.stem.lower():
            print(f"\n🔍 Detector detectado: {onnx_file.name}")
            model_workspace = min(workspace_gb + 1, 3)  # Max 3GB para detector
        elif 'w600k' in onnx_file.stem.lower() or 'r50' in onnx_file.stem.lower():
            print(f"\n🧠 Recognition model detectado: {onnx_file.name}")
        
        try:
            if build_engine(
                str(onnx_file),
                str(engine_file),
                fp16=fp16,
                workspace_gb=model_workspace,
                batch_size=1
            ):
                success_count += 1
        except Exception as e:
            print(f"\n✗ ERROR al convertir {onnx_file.name}:")
            print(f"  {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Conversión completada: {success_count}/{len(onnx_files)} exitosos")
    print(f"{'='*60}\n")
    
    return success_count == len(onnx_files)


def main():
    parser = argparse.ArgumentParser(
        description='Convertir modelos ONNX de Buffalo a TensorRT engines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Convertir todos los modelos en un directorio
  python convert_onnx_to_engine.py --input ./buffalo_l --output ./engines
  
  # Convertir un modelo específico
  python convert_onnx_to_engine.py --input det_10g.onnx --output det_10g.engine
  
  # Usar FP32 en vez de FP16 (más preciso pero más lento)
  python convert_onnx_to_engine.py --input ./buffalo_l --output ./engines --no-fp16
  
  # Reducir memoria de trabajo (para RAM limitada)
  python convert_onnx_to_engine.py --input ./buffalo_l --output ./engines --workspace 1
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Directorio con .onnx o archivo .onnx específico')
    parser.add_argument('--output', '-o', required=True,
                        help='Directorio de salida o archivo .engine de salida')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Deshabilitar FP16 (usar FP32)')
    parser.add_argument('--workspace', '-w', type=float, default=2.0,
                        help='Memoria de trabajo en GB (default: 2.0)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Verificar que existe el input
    if not input_path.exists():
        print(f"✗ ERROR: No existe {input_path}")
        return 1
    
    # Caso 1: Input es un directorio
    if input_path.is_dir():
        success = convert_buffalo_models(
            input_path,
            output_path,
            fp16=not args.no_fp16,
            workspace_gb=args.workspace
        )
        return 0 if success else 1
    
    # Caso 2: Input es un archivo .onnx específico
    elif input_path.suffix == '.onnx':
        # Si output es directorio, crear nombre automático
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            engine_path = output_path / f"{input_path.stem}.engine"
        else:
            engine_path = output_path
            engine_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = build_engine(
            str(input_path),
            str(engine_path),
            fp16=not args.no_fp16,
            workspace_gb=args.workspace,
            batch_size=1
        )
        return 0 if success else 1
    
    else:
        print(f"✗ ERROR: {input_path} debe ser un directorio o archivo .onnx")
        return 1


if __name__ == "__main__":
    sys.exit(main())