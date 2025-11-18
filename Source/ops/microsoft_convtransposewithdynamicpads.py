# ops/convtransposewithdynamicpads.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def convtransposewithdynamicpads_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    W = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [3, 2, 3, 3])  # C_out, C_in/group, kH, kW
    Pads = onnx.helper.make_tensor_value_info("Pads", onnx.TensorProto.INT64, [4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "ConvTransposeWithDynamicPads",
        inputs=["X", "W", "Pads"],
        outputs=["Y"],
        domain="com.microsoft",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        output_padding=[0, 0],
        strides=[2, 2]
    )

    graph = onnx.helper.make_graph(
        [node],
        "ConvTransposeWithDynamicPadsGraph",
        [X, W, Pads],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def convtransposewithdynamicpads_input_generator(session):
    X = np.random.randn(1, 3, 8, 8).astype(np.float32)
    W = np.random.randn(3, 2, 3, 3).astype(np.float32)
    Pads = np.array([1, 1, 1, 1], dtype=np.int64)  # [pad_top, pad_left, pad_bottom, pad_right]
    return {
        "X": X,
        "W": W,
        "Pads": Pads
    }

SpecialModelBuilders["com.microsoft.ConvTransposeWithDynamicPads"] = convtransposewithdynamicpads_model_builder
SpecialInputGenerators["com.microsoft.ConvTransposeWithDynamicPads"] = convtransposewithdynamicpads_input_generator
