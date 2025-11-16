import onnx
from onnx import numpy_helper

def fix_int64_to_int32(onnx_model_path, fixed_path):
    model = onnx.load(onnx_model_path)
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.INT64:
            arr = numpy_helper.to_array(tensor).astype('int32')
            tensor.CopyFrom(numpy_helper.from_array(arr, tensor.name))
    onnx.save(model, fixed_path)

fix_int64_to_int32("retinaface.onnx", "retinaface_fixed.onnx")
fix_int64_to_int32("arcface_r100.onnx", "arcface_r100_fixed.onnx")
