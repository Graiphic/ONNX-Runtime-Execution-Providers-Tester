# ops/gemm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gemm_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 4])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [4, 3])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [2, 3])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0
    )

    graph = onnx.helper.make_graph([node], "gemm_graph", [A, B, C], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def gemm_input_generator(session):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 3).astype(np.float32)
    C = np.random.rand(2, 3).astype(np.float32)
    return {
        "A": A,
        "B": B,
        "C": C
    }

SpecialModelBuilders["Gemm"] = gemm_model_builder
SpecialInputGenerators["Gemm"] = gemm_input_generator
