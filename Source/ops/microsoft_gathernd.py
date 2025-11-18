# ops/gathernd.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gathernd_model_builder(op_type, cfg=None):
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2])
    indices_info = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2, 2])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GatherND",
        inputs=["data", "indices"],
        outputs=["output"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "GatherNDGraph",
        [data_info, indices_info],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gathernd_input_generator(session):
    data = np.array([[0, 1], [2, 3]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    return {
        "data": data,
        "indices": indices
    }

SpecialModelBuilders["com.microsoft.GatherND"] = gathernd_model_builder
SpecialInputGenerators["com.microsoft.GatherND"] = gathernd_input_generator
