# ops/microsoft_packedattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def packedattention_model_builder(op_type, cfg=None):
    batch_size = 3
    max_seq_len = 4
    token_count = 7
    input_hidden_size = 16
    hidden_size = 16
    v_hidden_size = 16
    num_heads = 4

    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [token_count, input_hidden_size])
    weights = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [input_hidden_size, hidden_size * 2 + v_hidden_size])
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [hidden_size * 2 + v_hidden_size])
    token_offset = onnx.helper.make_tensor_value_info("token_offset", onnx.TensorProto.INT32, [batch_size, max_seq_len])
    cumulative_sequence_length = onnx.helper.make_tensor_value_info("cumulative_sequence_length", onnx.TensorProto.INT32, [batch_size + 1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "PackedAttention",
        inputs=["input", "weights", "bias", "token_offset", "cumulative_sequence_length"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        qkv_hidden_sizes=[hidden_size, hidden_size, v_hidden_size]
    )

    graph = onnx.helper.make_graph(
        [node],
        "PackedAttentionGraph",
        [input_tensor, weights, bias, token_offset, cumulative_sequence_length],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def packedattention_input_generator(session):
    batch_size = 3
    max_seq_len = 4
    token_count = 7
    input_hidden_size = 16
    hidden_size = 16
    v_hidden_size = 16

    # Exemple : séquences de tailles 1, 2, 4
    token_offset_data = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, -1],
        [7, 8, 9, 10]
    ], dtype=np.int32)

    cumulative_sequence_length_data = np.array([0, 1, 3, 7], dtype=np.int32)

    inputs = {
        "input": np.random.rand(token_count, input_hidden_size).astype(np.float32),
        "weights": np.random.rand(input_hidden_size, hidden_size * 2 + v_hidden_size).astype(np.float32),
        "bias": np.random.rand(hidden_size * 2 + v_hidden_size).astype(np.float32),
        "token_offset": token_offset_data,
        "cumulative_sequence_length": cumulative_sequence_length_data
    }
    return inputs

SpecialModelBuilders["com.microsoft.PackedAttention"] = packedattention_model_builder
SpecialInputGenerators["com.microsoft.PackedAttention"] = packedattention_input_generator
