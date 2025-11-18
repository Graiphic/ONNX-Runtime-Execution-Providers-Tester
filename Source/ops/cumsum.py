# ops/cumsum.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def cumsum_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    axis = onnx.helper.make_tensor_value_info("axis", TensorProto.INT32, [1])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "CumSum",
        inputs=["X", "axis"],
        outputs=["Y"],
        exclusive=0,
        reverse=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "cumsum_graph",
        [x, axis],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["CumSum"] = cumsum_model_builder

def cumsum_input_generator(session):
    x_info, axis_info = session.get_inputs()
    x = np.random.rand(3, 4).astype(np.float32)
    axis = np.array([1], dtype=np.int32)  # cumule le long des colonnes
    return {x_info.name: x, axis_info.name: axis}

SpecialInputGenerators["CumSum"] = cumsum_input_generator
