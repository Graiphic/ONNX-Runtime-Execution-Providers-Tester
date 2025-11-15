# ops/microsoft_qlinearmul.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearmul_model_builder(op_type, cfg=None):
    shape = [2, 3]

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, shape)
    A_scale = onnx.helper.make_tensor_value_info("A_scale", onnx.TensorProto.FLOAT, [])
    A_zero_point = onnx.helper.make_tensor_value_info("A_zero_point", onnx.TensorProto.UINT8, [])

    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, shape)
    B_scale = onnx.helper.make_tensor_value_info("B_scale", onnx.TensorProto.FLOAT, [])
    B_zero_point = onnx.helper.make_tensor_value_info("B_zero_point", onnx.TensorProto.UINT8, [])

    C_scale = onnx.helper.make_tensor_value_info("C_scale", onnx.TensorProto.FLOAT, [])
    C_zero_point = onnx.helper.make_tensor_value_info("C_zero_point", onnx.TensorProto.UINT8, [])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearMul",
        inputs=[
            "A", "A_scale", "A_zero_point",
            "B", "B_scale", "B_zero_point",
            "C_scale", "C_zero_point"
        ],
        outputs=["C"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearMulGraph",
        [A, A_scale, A_zero_point, B, B_scale, B_zero_point, C_scale, C_zero_point],
        [C]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearmul_input_generator(session):
    shape = [2, 3]
    return {
        "A": np.random.randint(0, 256, size=shape, dtype=np.uint8),
        "A_scale": np.array(0.02, dtype=np.float32),
        "A_zero_point": np.array(128, dtype=np.uint8),
        "B": np.random.randint(0, 256, size=shape, dtype=np.uint8),
        "B_scale": np.array(0.01, dtype=np.float32),
        "B_zero_point": np.array(128, dtype=np.uint8),
        "C_scale": np.array(0.05, dtype=np.float32),
        "C_zero_point": np.array(128, dtype=np.uint8)
    }

SpecialModelBuilders["com.microsoft.QLinearMul"] = qlinearmul_model_builder
SpecialInputGenerators["com.microsoft.QLinearMul"] = qlinearmul_input_generator
