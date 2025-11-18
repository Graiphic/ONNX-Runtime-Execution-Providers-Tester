# ops/flatten.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def flatten_model_builder(op_type, cfg=None):
    inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 3, 4])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Flatten",
        inputs=["input"],
        outputs=["output"],
        axis=1  # default axis is 1
    )

    graph = onnx.helper.make_graph([node], "flatten_graph", [inp], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def flatten_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {input_info.name: x}

SpecialModelBuilders["Flatten"] = flatten_model_builder
SpecialInputGenerators["Flatten"] = flatten_input_generator
