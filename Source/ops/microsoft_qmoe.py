# ops/microsoft_qmoe.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qmoe_model_builder(op_type, cfg=None):
    dtype_f = onnx.TensorProto.FLOAT16
    dtype_u = onnx.TensorProto.UINT8

    num_experts = 4
    hidden_size = 64
    inter_size = 256

    inputs = [
        onnx.helper.make_tensor_value_info("input", dtype_f, [2, hidden_size]),
        onnx.helper.make_tensor_value_info("router_probs", dtype_f, [2, num_experts]),
        onnx.helper.make_tensor_value_info("fc1_experts_weights", dtype_u, [num_experts, hidden_size, inter_size]),
        onnx.helper.make_tensor_value_info("fc1_scales", dtype_f, [num_experts, inter_size]),
        onnx.helper.make_tensor_value_info("fc1_experts_bias", dtype_f, [num_experts, inter_size]),
        onnx.helper.make_tensor_value_info("fc2_experts_weights", dtype_u, [num_experts, inter_size, hidden_size]),
        onnx.helper.make_tensor_value_info("fc2_scales", dtype_f, [num_experts, hidden_size]),
        onnx.helper.make_tensor_value_info("fc2_experts_bias", dtype_f, [num_experts, hidden_size])
    ]

    # Ajout des "" pour les 3 dernières entrées optionnelles (fc3)
    input_names = [i.name for i in inputs] + ["", "", ""]

    output = onnx.helper.make_tensor_value_info("output", dtype_f, None)

    node = onnx.helper.make_node(
        "QMoE",
        inputs=input_names,
        outputs=["output"],
        domain="com.microsoft",
        k=2,
        activation_type="relu",
        expert_weight_bits=8,
        normalize_routing_weights=1,
        use_sparse_mixer=0
    )

    graph = onnx.helper.make_graph([node], "QMoEGraph", inputs, [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model


def qmoe_input_generator(session):
    shape_map = {
        "input": (2, 64),
        "router_probs": (2, 4),
        "fc1_experts_weights": (4, 64, 256),
        "fc1_scales": (4, 256),
        "fc1_experts_bias": (4, 256),
        "fc2_experts_weights": (4, 256, 64),
        "fc2_scales": (4, 64),
        "fc2_experts_bias": (4, 64)
    }

    feed = {}
    for inp in session.get_inputs():
        shape = shape_map[inp.name]
        if inp.type == "tensor(uint8)":
            feed[inp.name] = np.random.randint(0, 16, size=shape, dtype=np.uint8)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float16)
    return feed


SpecialModelBuilders["com.microsoft.QMoE"] = qmoe_model_builder
SpecialInputGenerators["com.microsoft.QMoE"] = qmoe_input_generator
