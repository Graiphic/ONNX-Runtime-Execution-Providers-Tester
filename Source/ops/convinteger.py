# ops/convinteger.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def convinteger_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.UINT8, [1, 3, 32, 32])
    w = onnx.helper.make_tensor_value_info("W", TensorProto.UINT8, [6, 3, 3, 3])
    x_zp = onnx.helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, [])
    w_zp = onnx.helper.make_tensor_value_info("w_zero_point", TensorProto.UINT8, [])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.INT32, None)

    node = onnx.helper.make_node(
        "ConvInteger",
        inputs=["X", "W", "x_zero_point", "w_zero_point"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "convinteger_graph",
        [x, w, x_zp, w_zp],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["ConvInteger"] = convinteger_model_builder

def convinteger_input_generator(session):
    x_info, w_info, x_zp_info, w_zp_info = session.get_inputs()
    x = np.random.randint(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
    w = np.random.randint(0, 256, size=(6, 3, 3, 3), dtype=np.uint8)
    x_zero_point = np.array(128, dtype=np.uint8)
    w_zero_point = np.array(128, dtype=np.uint8)
    return {
        x_info.name: x,
        w_info.name: w,
        x_zp_info.name: x_zero_point,
        w_zp_info.name: w_zero_point
    }

SpecialInputGenerators["ConvInteger"] = convinteger_input_generator
