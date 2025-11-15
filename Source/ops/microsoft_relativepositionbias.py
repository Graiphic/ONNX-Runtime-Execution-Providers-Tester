# ops/microsoft_relativepositionbias.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def relativepositionbias_model_builder(op_type, cfg=None):
    num_buckets = 32
    num_heads = 2
    query_len = key_len = 4

    # Inputs
    bias = onnx.helper.make_tensor_value_info("bias_table", onnx.TensorProto.FLOAT, [num_buckets, num_heads])
    q_len = onnx.helper.make_tensor_value_info("query_length", onnx.TensorProto.INT64, [])
    k_len = onnx.helper.make_tensor_value_info("key_length", onnx.TensorProto.INT64, [])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, num_heads, query_len, key_len])

    # Constantes scalaires
    const_query = onnx.helper.make_tensor("query_length", onnx.TensorProto.INT64, [], [query_len])
    const_key = onnx.helper.make_tensor("key_length", onnx.TensorProto.INT64, [], [key_len])

    # Initializer pour bias_table (COL major = transposé dans NumPy)
    bias_table = np.random.randn(num_buckets, num_heads).astype(np.float32)
    const_bias = onnx.helper.make_tensor("bias_table", onnx.TensorProto.FLOAT, bias_table.shape, bias_table.flatten())

    node = onnx.helper.make_node(
        "RelativePositionBias",
        inputs=["bias_table", "query_length", "key_length"],
        outputs=["output"],
        domain="com.microsoft",
        max_distance=128,
        is_bidirectional=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "RelativePositionBiasGraph",
        [],
        [out],
        initializer=[const_bias, const_query, const_key]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def relativepositionbias_input_generator(session):
    return {}

SpecialModelBuilders["com.microsoft.RelativePositionBias"] = relativepositionbias_model_builder
SpecialInputGenerators["com.microsoft.RelativePositionBias"] = relativepositionbias_input_generator
