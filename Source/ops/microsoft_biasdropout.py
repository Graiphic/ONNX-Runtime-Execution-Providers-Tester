# ops/biasdropout.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def biasdropout_model_builder(op_type, cfg=None):
    shape = [2, 16, 16]  # data, bias, residual shape
    N, S, C = shape

    # Inputs
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, shape)
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [C])
    residual = onnx.helper.make_tensor_value_info("residual", onnx.TensorProto.FLOAT, shape)

    # Outputs
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
    mask = onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.BOOL, shape)

    node = onnx.helper.make_node(
        "BiasDropout",
        inputs=["data", "bias", "residual"],
        outputs=["output", "mask"],
        domain="com.microsoft"
        # pas besoin de seed ici, sauf pour tests reproductibles
    )

    graph = onnx.helper.make_graph(
        [node],
        "BiasDropoutGraph",
        [data, bias, residual],
        [output, mask]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def biasdropout_input_generator(session):
    N, S, C = 2, 16, 16
    data = np.random.randn(N, S, C).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)
    residual = np.random.randn(N, S, C).astype(np.float32)
    return {
        "data": data,
        "bias": bias,
        "residual": residual
    }

SpecialModelBuilders["com.microsoft.BiasDropout"] = biasdropout_model_builder
SpecialInputGenerators["com.microsoft.BiasDropout"] = biasdropout_input_generator
