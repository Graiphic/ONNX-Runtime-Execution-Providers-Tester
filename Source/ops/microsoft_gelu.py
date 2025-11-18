# ops/gelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gelu_model_builder(op_type, cfg=None):
    X_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Gelu",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "GeluGraph",
        [X_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gelu_input_generator(session):
    X = np.random.randn(2, 4).astype(np.float32)
    return {
        "X": X
    }

SpecialModelBuilders["com.microsoft.Gelu"] = gelu_model_builder
SpecialInputGenerators["com.microsoft.Gelu"] = gelu_input_generator
