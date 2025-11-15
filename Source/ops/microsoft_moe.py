# ops/microsoft_moe.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def moe_model_builder(op_type, cfg=None):
    num_rows = 4
    hidden_size = 16
    inter_size = 32
    num_experts = 4
    top_k = 1

    input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [num_rows, hidden_size])
    router = onnx.helper.make_tensor_value_info("router_probs", onnx.TensorProto.FLOAT, [num_rows, num_experts])
    fc1w = onnx.helper.make_tensor_value_info("fc1_experts_weights", onnx.TensorProto.FLOAT, [num_experts, hidden_size, inter_size])
    fc2w = onnx.helper.make_tensor_value_info("fc2_experts_weights", onnx.TensorProto.FLOAT, [num_experts, inter_size, hidden_size])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [num_rows, hidden_size])

    node = onnx.helper.make_node(
        "MoE",
        inputs=["input", "router_probs", "fc1_experts_weights", "", "fc2_experts_weights"],  # fc1 bias skipped
        outputs=["output"],
        domain="com.microsoft",
        k=top_k,
        activation_type="relu",
        normalize_routing_weights=1,
        use_sparse_mixer=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "MoEGraph",
        [input, router, fc1w, fc2w],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def moe_input_generator(session):
    num_rows = 4
    hidden_size = 16
    inter_size = 32
    num_experts = 4

    input = np.random.randn(num_rows, hidden_size).astype(np.float32)
    probs = np.random.rand(num_rows, num_experts).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    fc1_w = np.random.randn(num_experts, hidden_size, inter_size).astype(np.float32)
    fc2_w = np.random.randn(num_experts, inter_size, hidden_size).astype(np.float32)

    return {
        "input": input,
        "router_probs": probs,
        "fc1_experts_weights": fc1_w,
        "fc2_experts_weights": fc2_w
    }

SpecialModelBuilders["com.microsoft.MoE"] = moe_model_builder
SpecialInputGenerators["com.microsoft.MoE"] = moe_input_generator
