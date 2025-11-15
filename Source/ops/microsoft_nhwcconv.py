# ops/microsoft_nhwcconv.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def nhwcconv_model_builder(op_type, cfg=None):
    N, H, W, C_in, C_out, k = 1, 5, 5, 3, 4, 3

    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [N, H, W, C_in])
    Wt = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [C_out, C_in, k, k])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [N, H, W, C_out])

    node = onnx.helper.make_node(
        "NhwcConv",
        inputs=["X", "W"],
        outputs=["Y"],
        domain="com.microsoft",
        kernel_shape=[k, k],
        pads=[1, 1, 1, 1],  # top, left, bottom, right
        strides=[1, 1],
        group=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "NhwcConvGraph",
        [X, Wt],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def nhwcconv_input_generator(session):
    N, H, W, C_in, C_out, k = 1, 5, 5, 3, 4, 3
    X = np.random.randn(N, H, W, C_in).astype(np.float32)
    W = np.random.randn(C_out, C_in, k, k).astype(np.float32)
    return {"X": X, "W": W}

SpecialModelBuilders["com.microsoft.NhwcConv"] = nhwcconv_model_builder
SpecialInputGenerators["com.microsoft.NhwcConv"] = nhwcconv_input_generator
