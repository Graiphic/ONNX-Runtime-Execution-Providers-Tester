# ops/gemmfastgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gemmfastgelu_model_builder(op_type, cfg=None):
    X_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    W_info = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [4, 3])
    bias_info = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GemmFastGelu",
        inputs=["X", "W", "bias"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "GemmFastGeluGraph",
        [X_info, W_info, bias_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gemmfastgelu_input_generator(session):
    X = np.random.randn(2, 4).astype(np.float32)
    W = np.random.randn(4, 3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    return {
        "X": X,
        "W": W,
        "bias": bias
    }

SpecialModelBuilders["com.microsoft.GemmFastGelu"] = gemmfastgelu_model_builder
SpecialInputGenerators["com.microsoft.GemmFastGelu"] = gemmfastgelu_input_generator
