# ops/microsoft_qgemm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qgemm_model_builder(op_type, cfg=None):
    M, K, N = 2, 4, 3

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [M, K])
    a_scale = onnx.helper.make_tensor_value_info("a_scale", onnx.TensorProto.FLOAT, [])
    a_zero_point = onnx.helper.make_tensor_value_info("a_zero_point", onnx.TensorProto.UINT8, [])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [K, N])
    b_scale = onnx.helper.make_tensor_value_info("b_scale", onnx.TensorProto.FLOAT, [])
    b_zero_point = onnx.helper.make_tensor_value_info("b_zero_point", onnx.TensorProto.UINT8, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "QGemm",
        inputs=["A", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=1.0,
        transA=0,
        transB=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "QGemmGraph",
        [A, a_scale, a_zero_point, B, b_scale, b_zero_point],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qgemm_input_generator(session):
    M, K, N = 2, 4, 3

    inputs = {
        "A": np.random.randint(0, 255, size=(M, K), dtype=np.uint8),
        "a_scale": np.array(0.02, dtype=np.float32),
        "a_zero_point": np.array(128, dtype=np.uint8),
        "B": np.random.randint(0, 255, size=(K, N), dtype=np.uint8),
        "b_scale": np.array(0.01, dtype=np.float32),
        "b_zero_point": np.array(128, dtype=np.uint8)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QGemm"] = qgemm_model_builder
SpecialInputGenerators["com.microsoft.QGemm"] = qgemm_input_generator
