# ops/microsoft_qlinearconv.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearconv_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.UINT8, [1, 1, 5, 5])
    x_scale = onnx.helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zp = onnx.helper.make_tensor_value_info("x_zero_point", onnx.TensorProto.UINT8, [])

    W = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.UINT8, [1, 1, 3, 3])
    w_scale = onnx.helper.make_tensor_value_info("w_scale", onnx.TensorProto.FLOAT, [])
    w_zp = onnx.helper.make_tensor_value_info("w_zero_point", onnx.TensorProto.UINT8, [])

    y_scale = onnx.helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zp = onnx.helper.make_tensor_value_info("y_zero_point", onnx.TensorProto.UINT8, [])

    Y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearConv",
        inputs=["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point"],
        outputs=["y"],
        domain="com.microsoft",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        group=1,
        channels_last=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearConvGraph",
        [X, x_scale, x_zp, W, w_scale, w_zp, y_scale, y_zp],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearconv_input_generator(session):
    inputs = {
        "x": np.random.randint(0, 256, size=(1, 1, 5, 5), dtype=np.uint8),
        "x_scale": np.array(0.02, dtype=np.float32),
        "x_zero_point": np.array(128, dtype=np.uint8),
        "w": np.random.randint(0, 256, size=(1, 1, 3, 3), dtype=np.uint8),
        "w_scale": np.array(0.03, dtype=np.float32),
        "w_zero_point": np.array(128, dtype=np.uint8),
        "y_scale": np.array(0.05, dtype=np.float32),
        "y_zero_point": np.array(128, dtype=np.uint8)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QLinearConv"] = qlinearconv_model_builder
SpecialInputGenerators["com.microsoft.QLinearConv"] = qlinearconv_input_generator
