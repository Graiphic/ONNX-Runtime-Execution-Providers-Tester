# ops/expand.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def expand_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3])
    shape = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2])
    out = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Expand",
        inputs=["input", "shape"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph([node], "expand_graph", [data, shape], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def expand_input_generator(session):
    input_info, shape_info = session.get_inputs()
    x = np.random.rand(1, 3).astype(np.float32)
    shape = np.array([3, 3], dtype=np.int64)  # Expand to (3,3)
    return {input_info.name: x, shape_info.name: shape}

SpecialModelBuilders["Expand"] = expand_model_builder
SpecialInputGenerators["Expand"] = expand_input_generator
