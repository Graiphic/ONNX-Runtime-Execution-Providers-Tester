# ops/maxpool.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def maxpool_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "MaxPool",
        inputs=["X"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
        ceil_mode=0
    )

    graph = onnx.helper.make_graph([node], "maxpool_graph", [X], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def maxpool_input_generator(session):
    X = np.random.rand(1, 3, 16, 16).astype(np.float32)
    return {"X": X}

SpecialModelBuilders["MaxPool"] = maxpool_model_builder
SpecialInputGenerators["MaxPool"] = maxpool_input_generator
