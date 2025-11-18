# ops/det.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def det_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, [])

    node = onnx.helper.make_node(
        "Det",
        inputs=["X"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "det_graph",
        [x],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Det"] = det_model_builder

def det_input_generator(session):
    x_info = session.get_inputs()[0]
    shape = [d or 1 for d in x_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {x_info.name: x}

SpecialInputGenerators["Det"] = det_input_generator
