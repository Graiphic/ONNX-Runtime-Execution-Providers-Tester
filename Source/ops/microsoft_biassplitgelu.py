# ops/biassplitgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def biassplitgelu_model_builder(op_type, cfg=None):
    shape = [2, 64, 12]  # (N, S, D)
    N, S, D = shape

    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [D])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [N, S, D // 2])

    node = onnx.helper.make_node(
        "BiasSplitGelu",
        inputs=["X", "bias"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "BiasSplitGeluGraph",
        [X, bias],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def biassplitgelu_input_generator(session):
    N, S, D = 2, 64, 12
    X = np.random.randn(N, S, D).astype(np.float32)
    bias = np.random.randn(D).astype(np.float32)
    return {
        "X": X,
        "bias": bias
    }

SpecialModelBuilders["com.microsoft.BiasSplitGelu"] = biassplitgelu_model_builder
SpecialInputGenerators["com.microsoft.BiasSplitGelu"] = biassplitgelu_input_generator
