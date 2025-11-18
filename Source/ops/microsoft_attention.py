# ops/attention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def attention_model_builder(op_type, cfg=None):
    input_dim = 64
    num_heads = 4
    qkv_hidden_size = input_dim

    # Définition des entrées
    input_vi = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, input_dim])
    weights_vi = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [input_dim, 3 * qkv_hidden_size])
    bias_vi = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3 * qkv_hidden_size])
    output_vi = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, input_dim])

    # Création du nœud Attention
    attn_node = onnx.helper.make_node(
        "Attention",
        inputs=["input", "weights", "bias"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        qkv_hidden_sizes=[qkv_hidden_size] * 3,
        unidirectional=0,
        do_rotary=0
    )

    # Construction du graphe
    graph = onnx.helper.make_graph(
        [attn_node],
        "AttentionGraph",
        [input_vi, weights_vi, bias_vi],
        [output_vi]
    )

    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def attention_input_generator(session):
    inputs = session.get_inputs()
    input_shape = inputs[0].shape
    input_dim = input_shape[2]
    qkv_hidden_size = input_dim
    weights_shape = [input_dim, 3 * qkv_hidden_size]
    bias_shape = [3 * qkv_hidden_size]

    input_data = np.random.rand(*input_shape).astype(np.float32)
    weights = np.random.rand(*weights_shape).astype(np.float32)
    bias = np.random.rand(*bias_shape).astype(np.float32)

    return {
        "input": input_data,
        "weights": weights,
        "bias": bias
    }

SpecialModelBuilders["com.microsoft.Attention"] = attention_model_builder
SpecialInputGenerators["com.microsoft.Attention"] = attention_input_generator
