# ops/fastgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def fastgelu_model_builder(op_type, cfg=None):
    X_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    bias_info = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [4])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "FastGelu",
        inputs=["X", "bias"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "FastGeluGraph",
        [X_info, bias_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def fastgelu_input_generator(session):
    X = np.random.randn(2, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    return {
        "X": X,
        "bias": bias
    }

SpecialModelBuilders["com.microsoft.FastGelu"] = fastgelu_model_builder
SpecialInputGenerators["com.microsoft.FastGelu"] = fastgelu_input_generator
