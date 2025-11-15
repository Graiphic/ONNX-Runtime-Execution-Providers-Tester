# ops/microsoft_unfoldtensor.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def unfoldtensor_model_builder(op_type, cfg=None):
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10])  # vecteur 1D de 10 éléments
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)  # shape inférée

    node = helper.make_node(
        "UnfoldTensor",
        inputs=["input"],
        outputs=["output"],
        domain="com.microsoft",
        dim=0,
        size=4,
        step=2
    )

    graph = helper.make_graph(
        [node],
        "UnfoldTensorGraph",
        [X],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def unfoldtensor_input_generator(session):
    X = np.arange(10).astype(np.float32)
    return {
        "input": X
    }

SpecialModelBuilders["com.microsoft.UnfoldTensor"] = unfoldtensor_model_builder
SpecialInputGenerators["com.microsoft.UnfoldTensor"] = unfoldtensor_input_generator
