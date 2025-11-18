# ops/gemmfloat8.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gemmfloat8_model_builder(op_type, cfg=None):
    A_info = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 4])
    B_info = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [3, 4])
    C_info = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [2, 3])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GemmFloat8",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        domain="com.microsoft",
        transA=0,
        transB=1,
        alpha=1.0,
        beta=1.0,
        dtype=1,  # FLOAT
        activation="NONE"
    )

    graph = onnx.helper.make_graph(
        [node],
        "GemmFloat8Graph",
        [A_info, B_info, C_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gemmfloat8_input_generator(session):
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(3, 4).astype(np.float32)  # B transposée => B.T @ A
    C = np.random.randn(2, 3).astype(np.float32)
    return {
        "A": A,
        "B": B,
        "C": C
    }

SpecialModelBuilders["com.microsoft.GemmFloat8"] = gemmfloat8_model_builder
SpecialInputGenerators["com.microsoft.GemmFloat8"] = gemmfloat8_input_generator
