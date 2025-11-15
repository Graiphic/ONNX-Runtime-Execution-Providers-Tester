# ops/microsoft_qlinearwhere.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearwhere_model_builder(op_type, cfg=None):
    shape = [2, 3]

    condition = onnx.helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, shape)
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.UINT8, shape)
    x_scale = onnx.helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zero_point = onnx.helper.make_tensor_value_info("x_zero_point", onnx.TensorProto.UINT8, [])

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT8, shape)
    y_scale = onnx.helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zero_point = onnx.helper.make_tensor_value_info("y_zero_point", onnx.TensorProto.UINT8, [])

    z_scale = onnx.helper.make_tensor_value_info("z_scale", onnx.TensorProto.FLOAT, [])
    z_zero_point = onnx.helper.make_tensor_value_info("z_zero_point", onnx.TensorProto.UINT8, [])

    Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearWhere",
        inputs=[
            "condition", "X", "x_scale", "x_zero_point",
            "Y", "y_scale", "y_zero_point",
            "z_scale", "z_zero_point"
        ],
        outputs=["Z"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearWhereGraph",
        [
            condition, X, x_scale, x_zero_point,
            Y, y_scale, y_zero_point,
            z_scale, z_zero_point
        ],
        [Z]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearwhere_input_generator(session):
    shape = [2, 3]
    return {
        "condition": np.random.rand(*shape) > 0.5,
        "X": np.random.randint(0, 256, size=shape, dtype=np.uint8),
        "x_scale": np.array(0.02, dtype=np.float32),
        "x_zero_point": np.array(128, dtype=np.uint8),
        "Y": np.random.randint(0, 256, size=shape, dtype=np.uint8),
        "y_scale": np.array(0.01, dtype=np.float32),
        "y_zero_point": np.array(128, dtype=np.uint8),
        "z_scale": np.array(0.05, dtype=np.float32),
        "z_zero_point": np.array(128, dtype=np.uint8)
    }

SpecialModelBuilders["com.microsoft.QLinearWhere"] = qlinearwhere_model_builder
SpecialInputGenerators["com.microsoft.QLinearWhere"] = qlinearwhere_input_generator
