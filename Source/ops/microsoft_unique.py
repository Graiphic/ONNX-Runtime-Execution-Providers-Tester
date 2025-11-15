# ops/microsoft_unique.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def unique_model_builder(op_type, cfg=None):
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [6])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, None)
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, None)
    counts = helper.make_tensor_value_info("counts", TensorProto.INT64, None)

    node = helper.make_node(
        "Unique",
        inputs=["x"],
        outputs=["y", "idx", "counts"],
        domain="com.microsoft"
    )

    graph = helper.make_graph(
        [node],
        "UniqueGraph",
        [x],
        [y, idx, counts]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def unique_input_generator(session):
    x = np.array([2, 1, 1, 3, 4, 3], dtype=np.int32)
    return {
        "x": x
    }

SpecialModelBuilders["com.microsoft.Unique"] = unique_model_builder
SpecialInputGenerators["com.microsoft.Unique"] = unique_input_generator
