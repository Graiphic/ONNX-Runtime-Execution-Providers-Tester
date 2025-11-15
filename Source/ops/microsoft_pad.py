# ops/microsoft_pad.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def pad_model_builder(op_type, cfg=None):
    input_tensor = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 2])
    pads_tensor = onnx.helper.make_tensor_value_info("pads", onnx.TensorProto.INT64, [4])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Pad",
        inputs=["data", "pads"],
        outputs=["output"],
        domain="com.microsoft",
        mode="constant"
    )

    graph = onnx.helper.make_graph(
        [node],
        "PadGraph",
        [input_tensor, pads_tensor],
        [output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def pad_input_generator(session):
    data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], dtype=np.float32)
    pads = np.array([0, 2, 0, 0], dtype=np.int64)  # padding 2 zeros before dim=1
    return {
        "data": data,
        "pads": pads
    }

SpecialModelBuilders["com.microsoft.Pad"] = pad_model_builder
SpecialInputGenerators["com.microsoft.Pad"] = pad_input_generator
