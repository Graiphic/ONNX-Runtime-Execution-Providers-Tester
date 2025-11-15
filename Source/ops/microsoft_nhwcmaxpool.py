# ops/nhwcmaxpool.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def nhwcmaxpool_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.UINT8, [1, 16, 16, 3])
    Y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "NhwcMaxPool",
        inputs=["x"],
        outputs=["y"],
        domain="com.microsoft",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0]
    )

    graph = onnx.helper.make_graph(
        [node],
        "NhwcMaxPoolGraph",
        [X],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def nhwcmaxpool_input_generator(session):
    input_shape = [1, 16, 16, 3]
    X = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    return {"x": X}

SpecialModelBuilders["com.microsoft.NhwcMaxPool"] = nhwcmaxpool_model_builder
SpecialInputGenerators["com.microsoft.NhwcMaxPool"] = nhwcmaxpool_input_generator
