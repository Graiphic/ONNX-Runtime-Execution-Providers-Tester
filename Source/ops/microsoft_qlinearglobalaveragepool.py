# ops/microsoft_qlinearglobalaveragepool.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearglobalaveragepool_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.UINT8, [1, 3, 4, 4])
    x_scale = onnx.helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zp = onnx.helper.make_tensor_value_info("x_zero_point", onnx.TensorProto.UINT8, [])
    y_scale = onnx.helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zp = onnx.helper.make_tensor_value_info("y_zero_point", onnx.TensorProto.UINT8, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearGlobalAveragePool",
        inputs=["X", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        outputs=["Y"],
        domain="com.microsoft",
        channels_last=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearGlobalAveragePoolGraph",
        [X, x_scale, x_zp, y_scale, y_zp],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearglobalaveragepool_input_generator(session):
    inputs = {
        "X": np.random.randint(0, 256, size=(1, 3, 4, 4), dtype=np.uint8),
        "x_scale": np.array(0.02, dtype=np.float32),
        "x_zero_point": np.array(128, dtype=np.uint8),
        "y_scale": np.array(0.05, dtype=np.float32),
        "y_zero_point": np.array(128, dtype=np.uint8)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QLinearGlobalAveragePool"] = qlinearglobalaveragepool_model_builder
SpecialInputGenerators["com.microsoft.QLinearGlobalAveragePool"] = qlinearglobalaveragepool_input_generator
