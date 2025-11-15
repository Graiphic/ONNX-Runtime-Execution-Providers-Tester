# ops/microsoft_matmulnbits.py
import numpy as np
import onnx
import onnx.helper
import math
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def matmulnbits_model_builder(op_type, cfg=None):
    # Configuration : 4 bits, block_size=32
    K = 32
    N = 64
    bits = 4
    block_size = 32
    k_blocks = (K + block_size - 1) // block_size
    blob_size = (block_size * bits) // 8

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, K])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [N, k_blocks, blob_size])
    scales = onnx.helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [N, k_blocks])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, N])

    node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales", "", ""],  # zero_points, g_idx ignorés
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size
    )

    graph = onnx.helper.make_graph(
        [node],
        "MatMulNBitsGraph",
        [A, B, scales],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def matmulnbits_input_generator(session):
    K, N, bits, block_size = 32, 64, 4, 32
    k_blocks = (K + block_size - 1) // block_size
    blob_size = (block_size * bits) // 8

    A = np.random.randn(1, K).astype(np.float32)
    B = np.random.randint(0, 256, size=(N, k_blocks, blob_size), dtype=np.uint8)
    scales = np.random.rand(N, k_blocks).astype(np.float32)
    return {
        "A": A,
        "B": B,
        "scales": scales
    }

SpecialModelBuilders["com.microsoft.MatMulNBits"] = matmulnbits_model_builder
SpecialInputGenerators["com.microsoft.MatMulNBits"] = matmulnbits_input_generator
