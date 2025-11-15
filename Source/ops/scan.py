# ops/scan.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def scan_model_builder(op_type, cfg=None):
    # Sous-graphe : accumulation et passage d'une copie à la sortie scan
    subgraph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node("Add", ["state_in", "scan_in"], ["state_out"]),
            onnx.helper.make_node("Identity", ["state_out"], ["scan_out"])
        ],
        name="scan_body",
        inputs=[
            onnx.helper.make_tensor_value_info("state_in", TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info("scan_in", TensorProto.FLOAT, [])
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("state_out", TensorProto.FLOAT, []),
            onnx.helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [])
        ]
    )

    # Entrées du modèle global
    scan_input = onnx.helper.make_tensor_value_info("scan_input", TensorProto.FLOAT, [4])
    init_state = onnx.helper.make_tensor_value_info("init", TensorProto.FLOAT, [])
    final_state = onnx.helper.make_tensor_value_info("final", TensorProto.FLOAT, [])
    scan_output = onnx.helper.make_tensor_value_info("scan_output", TensorProto.FLOAT, [4])

    node = onnx.helper.make_node(
        "Scan",
        inputs=["init", "scan_input"],
        outputs=["final", "scan_output"],
        num_scan_inputs=1,
        body=subgraph
    )

    graph = onnx.helper.make_graph(
        [node],
        "scan_graph",
        inputs=[init_state, scan_input],
        outputs=[final_state, scan_output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def scan_input_generator(session):
    return {
        "init": np.array(0.0, dtype=np.float32),
        "scan_input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    }

SpecialModelBuilders["Scan"] = scan_model_builder
SpecialInputGenerators["Scan"] = scan_input_generator
