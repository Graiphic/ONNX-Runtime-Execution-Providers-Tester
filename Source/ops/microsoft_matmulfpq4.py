# ops/microsoft_matmulfpq4.py
import numpy as np
import onnx
import onnx.helper
import math
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def matmulfpq4_model_builder(op_type, cfg=None):
    # A @ B -> [1, K] x [K, N] = [1, N]
    K, N = 32, 64
    blk_quant_type = 0  # bloc de 32, pas d’offset

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, K])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [(K * N + 1) // 2])
    B_shape = onnx.helper.make_tensor_value_info("B_shape", onnx.TensorProto.INT64, [2])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, N])

    node = onnx.helper.make_node(
        "MatMulFpQ4",
        inputs=["A", "B", "B_shape"],
        outputs=["Y"],
        domain="com.microsoft",
        blk_quant_type=blk_quant_type
    )

    graph = onnx.helper.make_graph(
        [node],
        "MatMulFpQ4Graph",
        [A, B, B_shape],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def matmulfpq4_input_generator(session):
    K, N = 32, 64
    A = np.random.randn(1, K).astype(np.float32)
    B_blob = np.random.randint(0, 16, size=(K * N + 1) // 2, dtype=np.uint8)
    B_shape = np.array([K, N], dtype=np.int64)
    return {
        "A": A,
        "B": B_blob,
        "B_shape": B_shape
    }

SpecialModelBuilders["com.microsoft.MatMulFpQ4"] = matmulfpq4_model_builder
SpecialInputGenerators["com.microsoft.MatMulFpQ4"] = matmulfpq4_input_generator
