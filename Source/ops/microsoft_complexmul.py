# ops/complexmul.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def complexmul_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [None, 2])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [None, 2])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, None)  # shape inférée dynamiquement

    node = onnx.helper.make_node(
        "ComplexMul",
        inputs=["A", "B"],
        outputs=["C"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "ComplexMulGraph",
        [A, B],
        [C]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def complexmul_input_generator(session):
    shape = (8, 2)  # 8 complexes (real, imag)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    return {
        "A": A,
        "B": B
    }


SpecialModelBuilders["com.microsoft.ComplexMul"] = complexmul_model_builder
SpecialInputGenerators["com.microsoft.ComplexMul"] = complexmul_input_generator
