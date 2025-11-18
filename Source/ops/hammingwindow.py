# ops/hammingwindow.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def hammingwindow_model_builder(op_type, cfg=None):
    size = onnx.helper.make_tensor_value_info("size", TensorProto.INT64, [])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "HammingWindow",
        inputs=["size"],
        outputs=["Y"],
        output_datatype=1,  # FLOAT
        periodic=1          # fenêtre périodique
    )

    graph = onnx.helper.make_graph(
        [node],
        "hammingwindow_graph",
        [size],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["HammingWindow"] = hammingwindow_model_builder

def hammingwindow_input_generator(session):
    size_info = session.get_inputs()[0]
    size = np.array(64, dtype=np.int64)  # taille de la fenêtre
    return {size_info.name: size}

SpecialInputGenerators["HammingWindow"] = hammingwindow_input_generator
