# ops/gather_elements.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gather_elements_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2, 3])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["output"],
        axis=1  # Exemple : récupération le long de l'axe 1
    )

    graph = onnx.helper.make_graph([node], "gather_elements_graph", [data, indices], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def gather_elements_input_generator(session):
    data_info, indices_info = session.get_inputs()
    data = np.random.rand(2, 3).astype(np.float32)
    # Indices valides pour l'axe 1 (valeurs entre 0 et 2)
    indices = np.random.randint(0, 3, size=(2, 3), dtype=np.int64)
    return {data_info.name: data, indices_info.name: indices}

SpecialModelBuilders["GatherElements"] = gather_elements_model_builder
SpecialInputGenerators["GatherElements"] = gather_elements_input_generator
