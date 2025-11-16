#!/usr/bin/env python3
"""
python3 convert_to_tensorrt.py --input retinaface.onnx --output retinaface.engine
python3 convert_to_tensorrt.py --input ./buffalo_models --output ./engines

"""

import tensorrt as trt
import os
import sys
from pathlib import Path
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=2, batch_size=1):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    return True

def convert_models(input_path, output_path, fp16=True, workspace_gb=2):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix == ".onnx":
        engine_file = output_path / f"{input_path.stem}.engine" if output_path.is_dir() else output_path
        if engine_file.exists():
            print(f"Engine ya existe: {engine_file}")
            return True
        return build_engine(str(input_path), str(engine_file), fp16, workspace_gb)

    elif input_path.is_dir():
        onnx_files = list(input_path.glob("*.onnx"))
        if not onnx_files:
            print(f"No se encontraron archivos ONNX en {input_path}")
            return False
        success = True
        for onnx_file in onnx_files:
            engine_file = output_path / f"{onnx_file.stem}.engine"
            if engine_file.exists():
                print(f"Engine ya existe: {engine_file}")
                continue
            if not build_engine(str(onnx_file), str(engine_file), fp16, workspace_gb):
                print(f"Error convirtiendo {onnx_file}")
                success = False
        return success
    else:
        print(f"{input_path} no es un archivo ONNX ni un directorio válido")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convertir ONNX a TensorRT engine")
    parser.add_argument('--input', '-i', required=True, help="Archivo ONNX o directorio")
    parser.add_argument('--output', '-o', required=True, help="Archivo .engine o directorio de salida")
    parser.add_argument('--no-fp16', action='store_true', help="Deshabilitar FP16")
    parser.add_argument('--workspace', '-w', type=float, default=2.0, help="Memoria de trabajo en GB")
    args = parser.parse_args()

    fp16 = not args.no_fp16
    success = convert_models(args.input, args.output, fp16, args.workspace)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


