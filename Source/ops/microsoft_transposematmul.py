# ops/microsoft_transposematmul.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def transposematmul_model_builder(op_type, cfg=None):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = helper.make_node(
        "TransposeMatMul",
        inputs=["A", "B"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=1.0,
        transA=0,
        transB=0
    )

    graph = helper.make_graph(
        [node],
        "TransposeMatMulGraph",
        [A, B],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def transposematmul_input_generator(session):
    A = np.random.randn(2, 3).astype(np.float32)
    B = np.random.randn(3, 4).astype(np.float32)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["com.microsoft.TransposeMatMul"] = transposematmul_model_builder
SpecialInputGenerators["com.microsoft.TransposeMatMul"] = transposematmul_input_generator
