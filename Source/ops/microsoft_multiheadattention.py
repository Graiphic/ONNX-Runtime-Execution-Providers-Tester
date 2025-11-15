# ops/microsoft_multiheadattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def multiheadattention_model_builder(op_type, cfg=None):
    B, S, H = 2, 4, 32
    num_heads = 2

    query = onnx.helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [B, S, H])
    key   = onnx.helper.make_tensor_value_info("key", onnx.TensorProto.FLOAT, [B, S, H])
    value = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [B, S, H])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [B, S, H])

    node = onnx.helper.make_node(
        "MultiHeadAttention",
        inputs=["query", "key", "value", "", "", "", "", "", "", ""],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads
    )

    graph = onnx.helper.make_graph(
        [node],
        "MultiHeadAttentionGraph",
        [query, key, value],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def multiheadattention_input_generator(session):
    B, S, H = 2, 4, 32
    x = np.random.randn(B, S, H).astype(np.float32)
    return {"query": x, "key": x, "value": x}

SpecialModelBuilders["com.microsoft.MultiHeadAttention"] = multiheadattention_model_builder
SpecialInputGenerators["com.microsoft.MultiHeadAttention"] = multiheadattention_input_generator
