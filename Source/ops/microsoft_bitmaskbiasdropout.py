# ops/bitmaskbiasdropout.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def bitmaskbiasdropout_model_builder(op_type, cfg=None):
    shape = [2, 16, 16]  # (batch, seq, dim)
    N, S, C = shape

    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, shape)
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [C])
    residual = onnx.helper.make_tensor_value_info("residual", onnx.TensorProto.FLOAT, shape)

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
    mask = onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.UINT32, shape)

    node = onnx.helper.make_node(
        "BitmaskBiasDropout",
        inputs=["data", "bias", "residual"],
        outputs=["output", "mask"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "BitmaskBiasDropoutGraph",
        [data, bias, residual],
        [output, mask]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def bitmaskbiasdropout_input_generator(session):
    N, S, C = 2, 16, 16
    data = np.random.randn(N, S, C).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)
    residual = np.random.randn(N, S, C).astype(np.float32)
    return {
        "data": data,
        "bias": bias,
        "residual": residual
    }

SpecialModelBuilders["com.microsoft.BitmaskBiasDropout"] = bitmaskbiasdropout_model_builder
SpecialInputGenerators["com.microsoft.BitmaskBiasDropout"] = bitmaskbiasdropout_input_generator
