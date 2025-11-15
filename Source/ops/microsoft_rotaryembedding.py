# ops/microsoft_rotaryembedding.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def rotaryembedding_model_builder(op_type, cfg=None):
    B, S, H, num_heads = 2, 4, 8, 2
    head_size = H // num_heads
    rotary_dim = head_size
    dtype = onnx.TensorProto.FLOAT

    inputs = [
        onnx.helper.make_tensor_value_info("input", dtype, [B, num_heads, S, head_size]),
        onnx.helper.make_tensor_value_info("position_ids", onnx.TensorProto.INT64, [B, S]),
        onnx.helper.make_tensor_value_info("cos_cache", dtype, [S, rotary_dim // 2]),
        onnx.helper.make_tensor_value_info("sin_cache", dtype, [S, rotary_dim // 2]),
    ]

    output = onnx.helper.make_tensor_value_info("output", dtype, [B, num_heads, S, head_size])

    node = onnx.helper.make_node(
        "RotaryEmbedding",
        inputs=["input", "position_ids", "cos_cache", "sin_cache"],
        outputs=["output"],
        domain="com.microsoft",
        interleaved=0,
        is_packed_batching=0,
        num_heads=num_heads,
        rotary_embedding_dim=rotary_dim,
        scale=1.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "RotaryEmbeddingGraph",
        inputs,
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def rotaryembedding_input_generator(session):
    B, S, H, num_heads = 2, 4, 8, 2
    head_size = H // num_heads
    rotary_dim = head_size

    return {
        "input": np.random.rand(B, num_heads, S, head_size).astype(np.float32),
        "position_ids": np.tile(np.arange(S), (B, 1)).astype(np.int64),
        "cos_cache": np.random.rand(S, rotary_dim // 2).astype(np.float32),
        "sin_cache": np.random.rand(S, rotary_dim // 2).astype(np.float32),
    }

SpecialModelBuilders["com.microsoft.RotaryEmbedding"] = rotaryembedding_model_builder
SpecialInputGenerators["com.microsoft.RotaryEmbedding"] = rotaryembedding_input_generator
