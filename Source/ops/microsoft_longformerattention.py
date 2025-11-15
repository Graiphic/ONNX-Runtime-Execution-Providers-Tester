# ops/microsoft_longformerattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def longformerattention_model_builder(op_type, cfg=None):
    B, S, H, heads = 2, 8, 64, 4  # batch, seq_len, hidden_size, num_heads
    W = 2  # window size

    input     = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [B, S, H])
    weight    = onnx.helper.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, [H, 3 * H])
    bias      = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3 * H])
    mask      = onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.FLOAT, [B, S])
    g_weight  = onnx.helper.make_tensor_value_info("global_weight", onnx.TensorProto.FLOAT, [H, 3 * H])
    g_bias    = onnx.helper.make_tensor_value_info("global_bias", onnx.TensorProto.FLOAT, [3 * H])
    global_   = onnx.helper.make_tensor_value_info("global", onnx.TensorProto.INT32, [B, S])
    output    = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [B, S, H])

    node = onnx.helper.make_node(
        "LongformerAttention",
        inputs=["input", "weight", "bias", "mask", "global_weight", "global_bias", "global"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=heads,
        window=W
    )

    graph = onnx.helper.make_graph(
        [node],
        "LongformerAttentionGraph",
        [input, weight, bias, mask, g_weight, g_bias, global_],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def longformerattention_input_generator(session):
    B, S, H = 2, 8, 64
    X = np.random.randn(B, S, H).astype(np.float32)
    W = np.random.randn(H, 3 * H).astype(np.float32)
    b = np.random.randn(3 * H).astype(np.float32)
    M = np.zeros((B, S), dtype=np.float32)         # mask = 0 → pas masqué
    GW = np.random.randn(H, 3 * H).astype(np.float32)
    Gb = np.random.randn(3 * H).astype(np.float32)
    G = np.zeros((B, S), dtype=np.int32)           # global attention = 0 partout

    return {
        "input": X,
        "weight": W,
        "bias": b,
        "mask": M,
        "global_weight": GW,
        "global_bias": Gb,
        "global": G
    }

SpecialModelBuilders["com.microsoft.LongformerAttention"] = longformerattention_model_builder
SpecialInputGenerators["com.microsoft.LongformerAttention"] = longformerattention_input_generator
