# ops/bitmaskdropout.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def bitmaskdropout_model_builder(op_type, cfg=None):
    shape = [2, 16, 32]  # data shape

    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, shape)
    ratio = onnx.helper.make_tensor_value_info("ratio", onnx.TensorProto.FLOAT, [])
    training_mode = onnx.helper.make_tensor_value_info("training_mode", onnx.TensorProto.BOOL, [])

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)
    mask = onnx.helper.make_tensor_value_info("mask", onnx.TensorProto.UINT32, None)

    node = onnx.helper.make_node(
        "BitmaskDropout",
        inputs=["data", "ratio", "training_mode"],
        outputs=["output", "mask"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "BitmaskDropoutGraph",
        [data, ratio, training_mode],
        [output, mask]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def bitmaskdropout_input_generator(session):
    shape = [2, 16, 32]
    data = np.random.randn(*shape).astype(np.float32)
    ratio = np.array(0.5, dtype=np.float32)
    training_mode = np.array(True, dtype=bool)

    return {
        "data": data,
        "ratio": ratio,
        "training_mode": training_mode
    }

SpecialModelBuilders["com.microsoft.BitmaskDropout"] = bitmaskdropout_model_builder
SpecialInputGenerators["com.microsoft.BitmaskDropout"] = bitmaskdropout_input_generator
