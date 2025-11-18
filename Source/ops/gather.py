# ops/gather.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gather_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [4, 5])
    indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Gather",
        inputs=["data", "indices"],
        outputs=["output"],
        axis=0
    )

    graph = onnx.helper.make_graph([node], "gather_graph", [data, indices], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def gather_input_generator(session):
    data_info, indices_info = session.get_inputs()
    data = np.random.rand(4, 5).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2,), dtype=np.int64)  # Axis = 0
    return {data_info.name: data, indices_info.name: indices}

SpecialModelBuilders["Gather"] = gather_model_builder
SpecialInputGenerators["Gather"] = gather_input_generator
