# ops/microsoft_maxpoolwithmask.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def maxpoolwithmask_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    M = onnx.helper.make_tensor_value_info("M", onnx.TensorProto.INT32, [1, 1, 4, 4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "MaxpoolWithMask",
        inputs=["X", "M"],
        outputs=["Y"],
        domain="com.microsoft",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
        storage_order=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "MaxpoolWithMaskGraph",
        [X, M],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def maxpoolwithmask_input_generator(session):
    X = np.random.rand(1, 1, 4, 4).astype(np.float32)
    M = np.ones((1, 1, 4, 4), dtype=np.int32)
    return {"X": X, "M": M}

SpecialModelBuilders["com.microsoft.MaxpoolWithMask"] = maxpoolwithmask_model_builder
SpecialInputGenerators["com.microsoft.MaxpoolWithMask"] = maxpoolwithmask_input_generator
