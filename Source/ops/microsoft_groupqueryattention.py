import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def groupqueryattention_model_builder(op_type, cfg=None):
    B, S = 2, 4
    q_heads = 4
    kv_heads = 4
    head_size = 16
    H = q_heads * head_size    # = 64
    H_kv = kv_heads * head_size  # aussi = 64, pour rester cohérent

    # Déclaration des infos de tenseur
    query  = onnx.helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [B, S, H])
    key    = onnx.helper.make_tensor_value_info("key",   onnx.TensorProto.FLOAT, [B, S, H_kv])
    value  = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [B, S, H_kv])
    seqlens_k = onnx.helper.make_tensor_value_info("seqlens_k", onnx.TensorProto.INT32, [B])
    total_len = onnx.helper.make_tensor_value_info("total_sequence_length", onnx.TensorProto.INT32, [1])

    # Sorties
    output = onnx.helper.make_tensor_value_info("output",        onnx.TensorProto.FLOAT, [B, S,   H])
    present_key = onnx.helper.make_tensor_value_info("present_key",  onnx.TensorProto.FLOAT, None)
    present_value = onnx.helper.make_tensor_value_info("present_value",onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "query", "key", "value", "", "",  # on donne key/value, pas de packed QKV
            "seqlens_k",
            "total_sequence_length",
            "", "", "", ""                     # cos_cache, sin_cache, position_ids, attention_bias
        ],
        outputs=["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=q_heads,
        kv_num_heads=kv_heads
    )

    graph = onnx.helper.make_graph(
        [node],
        "GroupQueryAttentionGraph",
        [query, key, value, seqlens_k, total_len],
        [output, present_key, present_value]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def groupqueryattention_input_generator(session):
    B, S, H = 2, 4, 64
    query = np.random.randn(B, S, H).astype(np.float32)
    key   = np.random.randn(B, S, H).astype(np.float32)  # même hidden_size que query
    value = np.random.randn(B, S, H).astype(np.float32)

    seqlens_k = np.array([S] * B, dtype=np.int32)
    total_sequence_length = np.array([S], dtype=np.int32)

    return {
        "query": query,
        "key":   key,
        "value": value,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_sequence_length
    }

SpecialModelBuilders["com.microsoft.GroupQueryAttention"]    = groupqueryattention_model_builder
SpecialInputGenerators["com.microsoft.GroupQueryAttention"] = groupqueryattention_input_generator
