# ops/fusedconv.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def fusedconv_model_builder(op_type, cfg=None):
    X_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 5, 5])
    W_info = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [2, 3, 3, 3])
    B_info = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [2])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "FusedConv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        domain="com.microsoft",
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        kernel_shape=[3, 3],
        group=1,
        activation="Relu"
    )

    graph = onnx.helper.make_graph(
        [node],
        "FusedConvGraph",
        [X_info, W_info, B_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def fusedconv_input_generator(session):
    X = np.random.randn(1, 3, 5, 5).astype(np.float32)
    W = np.random.randn(2, 3, 3, 3).astype(np.float32)
    B = np.random.randn(2).astype(np.float32)
    return {
        "X": X,
        "W": W,
        "B": B
    }

SpecialModelBuilders["com.microsoft.FusedConv"] = fusedconv_model_builder
SpecialInputGenerators["com.microsoft.FusedConv"] = fusedconv_input_generator
