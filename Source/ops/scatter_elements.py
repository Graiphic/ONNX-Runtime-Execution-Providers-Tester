# ops/scatter_elements.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def scatter_elements_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2, 3])
    updates = onnx.helper.make_tensor_value_info("updates", onnx.TensorProto.FLOAT, [2, 3])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "ScatterElements",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
        axis=1
    )

    graph = onnx.helper.make_graph([node], "scatter_elements_graph", [data, indices, updates], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def scatter_elements_input_generator(session):
    data_info, indices_info, updates_info = session.get_inputs()
    data = np.zeros((2, 3), dtype=np.float32)
    indices = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    updates = np.random.rand(2, 3).astype(np.float32)
    return {
        data_info.name: data,
        indices_info.name: indices,
        updates_info.name: updates
    }

SpecialModelBuilders["ScatterElements"] = scatter_elements_model_builder
SpecialInputGenerators["ScatterElements"] = scatter_elements_input_generator
