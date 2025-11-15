# ops/microsoft_matmulbnb4.py
import numpy as np
import onnx
import onnx.helper
import math
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def matmulbnb4_model_builder(op_type, cfg=None):
    # A: [batch, K]
    # B: quantized, shape [(N*K + 1)//2]
    # absmax: shape [(N*K + block_size - 1)//block_size]
    K = 32
    N = 64
    block_size = 32
    quant_type = 0  # FP4

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, K])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [(N * K + 1) // 2])
    absmax = onnx.helper.make_tensor_value_info("absmax", onnx.TensorProto.FLOAT, [(N * K + block_size - 1) // block_size])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, N])

    node = onnx.helper.make_node(
        "MatMulBnb4",
        inputs=["A", "B", "absmax"],
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        block_size=block_size,
        quant_type=quant_type,
        transB=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "MatMulBnb4Graph",
        [A, B, absmax],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def matmulbnb4_input_generator(session):
    K, N = 32, 64
    block_size = 32
    A = np.random.randn(1, K).astype(np.float32)
    quant_bytes = (N * K + 1) // 2
    scale_entries = (N * K + block_size - 1) // block_size
    B = np.random.randint(0, 16, size=quant_bytes, dtype=np.uint8)
    absmax = np.random.rand(scale_entries).astype(np.float32)
    return {
        "A": A,
        "B": B,
        "absmax": absmax
    }

SpecialModelBuilders["com.microsoft.MatMulBnb4"] = matmulbnb4_model_builder
SpecialInputGenerators["com.microsoft.MatMulBnb4"] = matmulbnb4_input_generator
