# ops/range.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def range_model_builder(op_type, cfg=None):
    start = onnx.helper.make_tensor_value_info("start", TensorProto.FLOAT, [])
    limit = onnx.helper.make_tensor_value_info("limit", TensorProto.FLOAT, [])
    delta = onnx.helper.make_tensor_value_info("delta", TensorProto.FLOAT, [])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Range",
        inputs=["start", "limit", "delta"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "range_graph",
        inputs=[start, limit, delta],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def range_input_generator(session):
    return {
        "start": np.array(0.0, dtype=np.float32),
        "limit": np.array(5.0, dtype=np.float32),
        "delta": np.array(1.0, dtype=np.float32)
    }

SpecialModelBuilders["Range"] = range_model_builder
SpecialInputGenerators["Range"] = range_input_generator
