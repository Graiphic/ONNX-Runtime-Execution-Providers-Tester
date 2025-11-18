# ops/einsum.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def einsum_model_builder(op_type, cfg=None):
    a = onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    b = onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Einsum",
        inputs=["A", "B"],
        outputs=["Y"],
        equation="ij,jk->ik"
    )

    graph = onnx.helper.make_graph(
        [node],
        "einsum_graph",
        [a, b],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Einsum"] = einsum_model_builder

def einsum_input_generator(session):
    a_info, b_info = session.get_inputs()
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    return {
        a_info.name: a,
        b_info.name: b
    }

SpecialInputGenerators["Einsum"] = einsum_input_generator
