# ops/optional.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def optional_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    optional_val = onnx.helper.make_node("Optional", ["X"], ["opt"])

    has_val = onnx.helper.make_node("OptionalHasElement", ["opt"], ["has"])
    get_val = onnx.helper.make_node("OptionalGetElement", ["opt"], ["Y"])

    has = onnx.helper.make_tensor_value_info("has", TensorProto.BOOL, [])

    graph = onnx.helper.make_graph(
        [optional_val, has_val, get_val],
        "optional_graph",
        inputs=[x],
        outputs=[has, y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def optional_input_generator(session):
    return {"X": np.array([3.14, 2.71], dtype=np.float32)}

SpecialModelBuilders["Optional"] = optional_model_builder
SpecialInputGenerators["Optional"] = optional_input_generator
