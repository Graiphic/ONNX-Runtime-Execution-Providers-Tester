# ops/gatedrelativepositionbias.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gatedrelativepositionbias_model_builder(op_type, cfg=None):
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_size = 8
    D = 16  # D divisible by 2

    q_shape = [batch_size, seq_len, num_heads * head_size]

    query_layer_info = onnx.helper.make_tensor_value_info("query_layer", onnx.TensorProto.FLOAT, q_shape)
    query_bias_info = onnx.helper.make_tensor_value_info("query_bias", onnx.TensorProto.FLOAT, [num_heads * head_size])
    rel_pos_info = onnx.helper.make_tensor_value_info("rel_pos", onnx.TensorProto.FLOAT, [1, num_heads, seq_len, seq_len])
    weight_info = onnx.helper.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, [head_size, D])
    bias_info = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [D])
    eco_a_info = onnx.helper.make_tensor_value_info("eco_a", onnx.TensorProto.FLOAT, [1, num_heads, 1, 1])
    
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GatedRelativePositionBias",
        inputs=["query_layer", "query_bias", "rel_pos", "weight", "bias", "eco_a"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads
    )

    graph = onnx.helper.make_graph(
        [node],
        "GatedRelativePositionBiasGraph",
        [
            query_layer_info,
            query_bias_info,
            rel_pos_info,
            weight_info,
            bias_info,
            eco_a_info
        ],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gatedrelativepositionbias_input_generator(session):
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_size = 8
    D = 16

    query_layer = np.random.randn(batch_size, seq_len, num_heads * head_size).astype(np.float32)
    query_bias = np.random.randn(num_heads * head_size).astype(np.float32)
    rel_pos = np.random.randn(1, num_heads, seq_len, seq_len).astype(np.float32)
    weight = np.random.randn(head_size, D).astype(np.float32)
    bias = np.random.randn(D).astype(np.float32)
    eco_a = np.random.randn(1, num_heads, 1, 1).astype(np.float32)

    return {
        "query_layer": query_layer,
        "query_bias": query_bias,
        "rel_pos": rel_pos,
        "weight": weight,
        "bias": bias,
        "eco_a": eco_a
    }

SpecialModelBuilders["com.microsoft.GatedRelativePositionBias"] = gatedrelativepositionbias_model_builder
SpecialInputGenerators["com.microsoft.GatedRelativePositionBias"] = gatedrelativepositionbias_input_generator
