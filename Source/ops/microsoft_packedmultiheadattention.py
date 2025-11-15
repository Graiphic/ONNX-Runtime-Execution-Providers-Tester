# ops/microsoft_packedmultiheadattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def packedmha_model_builder(op_type, cfg=None):
    batch_size = 3
    max_seq_len = 4
    token_count = 7
    hidden_size = 16
    v_hidden_size = 16
    num_heads = 4

    # Inputs
    query = onnx.helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [token_count, hidden_size])
    key = onnx.helper.make_tensor_value_info("key", onnx.TensorProto.FLOAT, [token_count, hidden_size])
    value = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [token_count, v_hidden_size])
    token_offset = onnx.helper.make_tensor_value_info("token_offset", onnx.TensorProto.INT32, [batch_size, max_seq_len])
    cumulative_sequence_length = onnx.helper.make_tensor_value_info("cumulative_sequence_length", onnx.TensorProto.INT32, [batch_size + 1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "PackedMultiHeadAttention",
        inputs=["query", "key", "value", "", "token_offset", "cumulative_sequence_length"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads
    )

    graph = onnx.helper.make_graph(
        [node],
        "PackedMhaGraph",
        [query, key, value, token_offset, cumulative_sequence_length],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def packedmha_input_generator(session):
    batch_size = 3
    max_seq_len = 4
    token_count = 7
    hidden_size = 16
    v_hidden_size = 16

    inputs = {
        "query": np.random.rand(token_count, hidden_size).astype(np.float32),
        "key": np.random.rand(token_count, hidden_size).astype(np.float32),
        "value": np.random.rand(token_count, v_hidden_size).astype(np.float32),
        "token_offset": np.array([
            [0, 1, 2, 3],
            [4, 5, 6, -1],
            [7, 8, 9, 10]
        ], dtype=np.int32),
        "cumulative_sequence_length": np.array([0, 1, 3, 7], dtype=np.int32)
    }
    return inputs

SpecialModelBuilders["com.microsoft.PackedMultiHeadAttention"] = packedmha_model_builder
SpecialInputGenerators["com.microsoft.PackedMultiHeadAttention"] = packedmha_input_generator
