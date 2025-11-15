# ops/prelu.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def prelu_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    slope = onnx.helper.make_tensor_value_info("slope", TensorProto.FLOAT, [1, 3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "PRelu",
        inputs=["X", "slope"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "prelu_graph",
        inputs=[x, slope],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def prelu_input_generator(session):
    x = np.random.randn(2, 3).astype(np.float32)
    slope = np.random.rand(1, 3).astype(np.float32)
    return {"X": x, "slope": slope}

SpecialModelBuilders["PRelu"] = prelu_model_builder
SpecialInputGenerators["PRelu"] = prelu_input_generator
