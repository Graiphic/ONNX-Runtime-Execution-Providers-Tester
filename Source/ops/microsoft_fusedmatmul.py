# ops/fusedmatmul.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def fusedmatmul_model_builder(op_type, cfg=None):
    A_info = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 4])
    B_info = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [4, 3])
    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "FusedMatMul",
        inputs=["A", "B"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=1.0,
        transA=0,
        transB=0,
        transBatchA=0,
        transBatchB=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "FusedMatMulGraph",
        [A_info, B_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def fusedmatmul_input_generator(session):
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(4, 3).astype(np.float32)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["com.microsoft.FusedMatMul"] = fusedmatmul_model_builder
SpecialInputGenerators["com.microsoft.FusedMatMul"] = fusedmatmul_input_generator
