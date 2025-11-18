# ops/decoderattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def decoderattention_model_builder(op_type, cfg=None):
    num_heads = 4
    head_size = 16
    hidden_size = num_heads * head_size

    query = onnx.helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [None, None, hidden_size])
    key = onnx.helper.make_tensor_value_info("key", onnx.TensorProto.FLOAT, [None, None, hidden_size])
    q_weight = onnx.helper.make_tensor_value_info("q_weight", onnx.TensorProto.FLOAT, [hidden_size, hidden_size])
    kv_weight = onnx.helper.make_tensor_value_info("kv_weight", onnx.TensorProto.FLOAT, [hidden_size, 2 * hidden_size])
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3 * hidden_size])
    static_kv = onnx.helper.make_tensor_value_info("static_kv", onnx.TensorProto.BOOL, [])
    use_past = onnx.helper.make_tensor_value_info("use_past", onnx.TensorProto.BOOL, [])
    has_layer_state = onnx.helper.make_tensor_value_info("has_layer_state", onnx.TensorProto.BOOL, [])
    has_key_padding_mask = onnx.helper.make_tensor_value_info("has_key_padding_mask", onnx.TensorProto.BOOL, [])

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DecoderAttention",
        inputs=[
            "query", "key", "q_weight", "kv_weight", "bias",
            "", "", "",  # placeholders pour key_padding_mask, key_cache, value_cache
            "static_kv", "use_past", "has_layer_state", "has_key_padding_mask"
        ],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        mask_filter_value=-10000.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "DecoderAttentionGraph",
        [
            query, key, q_weight, kv_weight, bias,
            static_kv, use_past, has_layer_state, has_key_padding_mask
        ],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def decoderattention_input_generator(session):
    seq_len, total_seq_len, batch_size, num_heads, head_size = 5, 10, 2, 4, 16
    hidden_size = num_heads * head_size

    return {
        "query": np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32),
        "key": np.random.randn(total_seq_len, batch_size, hidden_size).astype(np.float32),
        "q_weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "kv_weight": np.random.randn(hidden_size, 2 * hidden_size).astype(np.float32),
        "bias": np.random.randn(3 * hidden_size).astype(np.float32),
        "static_kv": np.array(False, dtype=bool),
        "use_past": np.array(False, dtype=bool),
        "has_layer_state": np.array(False, dtype=bool),
        "has_key_padding_mask": np.array(False, dtype=bool)
    }

SpecialModelBuilders["com.microsoft.DecoderAttention"] = decoderattention_model_builder
SpecialInputGenerators["com.microsoft.DecoderAttention"] = decoderattention_input_generator
