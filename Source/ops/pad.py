# ops/pad.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def pad_model_builder(op_type, cfg=None):
    inp = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    pads = onnx.helper.make_tensor_value_info("pads", onnx.TensorProto.INT64, [4])  # 2*rank
    value = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Pad",
        inputs=["data", "pads", "value"],
        outputs=["output"],
        mode="constant"
    )

    graph = onnx.helper.make_graph([node], "pad_graph", [inp, pads, value], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def pad_input_generator(session):
    data_info, pads_info, value_info = session.get_inputs()
    data = np.random.rand(2, 3).astype(np.float32)
    pads = np.array([1, 0, 1, 0], dtype=np.int64)  # pad before/after on axis 0 and 1
    value = np.array(0.5, dtype=np.float32)
    return {
        data_info.name: data,
        pads_info.name: pads,
        value_info.name: value
    }

SpecialModelBuilders["Pad"] = pad_model_builder
SpecialInputGenerators["Pad"] = pad_input_generator
