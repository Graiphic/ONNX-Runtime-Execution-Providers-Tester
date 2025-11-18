# ops/cdist.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def cdist_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [None, None])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [None, None])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, None)  # shape inférée dynamiquement

    node = onnx.helper.make_node(
        "CDist",
        inputs=["A", "B"],
        outputs=["C"],
        domain="com.microsoft",
        metric="euclidean"
    )

    graph = onnx.helper.make_graph(
        [node],
        "CDistGraph",
        [A, B],
        [C]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def cdist_input_generator(session):
    M, K, N = 10, 8, 5
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["com.microsoft.CDist"] = cdist_model_builder
SpecialInputGenerators["com.microsoft.CDist"] = cdist_input_generator
