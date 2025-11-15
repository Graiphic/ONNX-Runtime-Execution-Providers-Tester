# ops/microsoft_trilu.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def trilu_model_builder(op_type, cfg=None):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 3])  # batch de 2 matrices 3x3
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 3])

    node = helper.make_node(
        "Trilu",
        inputs=["X"],  # pas de k
        outputs=["Y"],
        domain="com.microsoft",
        upper=1  # upper triangle conservé
    )

    graph = helper.make_graph(
        [node],
        "TriluGraph",
        [X],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def trilu_input_generator(session):
    X = np.random.randn(2, 3, 3).astype(np.float32)
    return {
        "X": X
    }

SpecialModelBuilders["com.microsoft.Trilu"] = trilu_model_builder
SpecialInputGenerators["com.microsoft.Trilu"] = trilu_input_generator
