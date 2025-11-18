# ops/convtranspose.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def convtranspose_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 6, 16, 16])  # NCHW
    w = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, [6, 3, 5, 5])   # C x M x H x W (note: flipped from Conv)
    b = onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, [3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "ConvTranspose",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[5, 5],
        strides=[2, 2],
        pads=[2, 2, 2, 2],
        dilations=[1, 1],
        group=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "convtranspose_graph",
        [x, w, b],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["ConvTranspose"] = convtranspose_model_builder

def convtranspose_input_generator(session):
    x_info, w_info, b_info = session.get_inputs()
    x = np.random.rand(1, 6, 16, 16).astype(np.float32)
    w = np.random.randn(6, 3, 5, 5).astype(np.float32) * 0.1  # C (in), M (out), H, W
    b = np.random.randn(3).astype(np.float32) * 0.01
    return {
        x_info.name: x,
        w_info.name: w,
        b_info.name: b
    }

SpecialInputGenerators["ConvTranspose"] = convtranspose_input_generator
