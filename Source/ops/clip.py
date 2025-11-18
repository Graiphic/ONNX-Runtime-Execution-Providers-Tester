# ops/clip.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def clip_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 2])
    min_val = onnx.helper.make_tensor_value_info("min", onnx.TensorProto.FLOAT, [1])
    max_val = onnx.helper.make_tensor_value_info("max", onnx.TensorProto.FLOAT, [1])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Clip",
        inputs=["X", "min", "max"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "clip_graph",
        [x, min_val, max_val],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]  # >= opset 11
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Clip"] = clip_model_builder

def clip_input_generator(session):
    x_info, min_info, max_info = session.get_inputs()
    shape = [d or 1 for d in x_info.shape]
    x = np.random.randn(*shape).astype(np.float32) * 10
    min_val = np.array([0.0], dtype=np.float32)
    max_val = np.array([6.0], dtype=np.float32)
    return {
        x_info.name: x,
        min_info.name: min_val,
        max_info.name: max_val
    }

SpecialInputGenerators["Clip"] = clip_input_generator
