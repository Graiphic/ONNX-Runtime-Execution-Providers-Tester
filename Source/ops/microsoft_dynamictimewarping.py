# ops/dynamictimewarping.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dynamictimewarping_model_builder(op_type, cfg=None):
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 5, 6])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT32, None)

    node = onnx.helper.make_node(
        "DynamicTimeWarping",
        inputs=["input"],
        outputs=["output"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "DynamicTimeWarpingGraph",
        [input_info],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def dynamictimewarping_input_generator(session):
    cost_matrix = np.random.rand(1, 5, 6).astype(np.float32)
    return {
        "input": cost_matrix
    }

SpecialModelBuilders["com.microsoft.DynamicTimeWarping"] = dynamictimewarping_model_builder
SpecialInputGenerators["com.microsoft.DynamicTimeWarping"] = dynamictimewarping_input_generator
