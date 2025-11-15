# ops/microsoft_qlinearsoftmax.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearsoftmax_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.UINT8, [2, 4])
    X_scale = onnx.helper.make_tensor_value_info("X_scale", onnx.TensorProto.FLOAT, [])
    X_zp = onnx.helper.make_tensor_value_info("x_zero_point", onnx.TensorProto.UINT8, [])
    Y_scale = onnx.helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    Y_zp = onnx.helper.make_tensor_value_info("y_zero_point", onnx.TensorProto.UINT8, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearSoftmax",
        inputs=["X", "X_scale", "x_zero_point", "y_scale", "y_zero_point"],
        outputs=["Y"],
        domain="com.microsoft",
        axis=1,
        opset=13
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearSoftmaxGraph",
        [X, X_scale, X_zp, Y_scale, Y_zp],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearsoftmax_input_generator(session):
    return {
        "X": np.random.randint(0, 256, size=(2, 4), dtype=np.uint8),
        "X_scale": np.array(0.02, dtype=np.float32),
        "x_zero_point": np.array(128, dtype=np.uint8),
        "y_scale": np.array(0.01, dtype=np.float32),
        "y_zero_point": np.array(128, dtype=np.uint8)
    }

SpecialModelBuilders["com.microsoft.QLinearSoftmax"] = qlinearsoftmax_model_builder
SpecialInputGenerators["com.microsoft.QLinearSoftmax"] = qlinearsoftmax_input_generator
