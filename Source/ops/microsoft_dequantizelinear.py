# ops/dequantizelinear.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dequantizelinear_model_builder(op_type, cfg=None):
    x_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.UINT8, [2, 3])
    x_scale_info = onnx.helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zero_point_info = onnx.helper.make_tensor_value_info("x_zero_point", onnx.TensorProto.UINT8, [])

    y_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "DequantizeLinearGraph",
        [x_info, x_scale_info, x_zero_point_info],
        [y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def dequantizelinear_input_generator(session):
    shape = (2, 3)
    x = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    x_scale = np.array(0.05, dtype=np.float32)
    x_zero_point = np.array(128, dtype=np.uint8)

    return {
        "x": x,
        "x_scale": x_scale,
        "x_zero_point": x_zero_point
    }

SpecialModelBuilders["com.microsoft.DequantizeLinear"] = dequantizelinear_model_builder
SpecialInputGenerators["com.microsoft.DequantizeLinear"] = dequantizelinear_input_generator
