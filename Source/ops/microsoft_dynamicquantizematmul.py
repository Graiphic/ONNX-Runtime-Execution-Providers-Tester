# ops/dynamicquantizematmul.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dynamicquantizematmul_model_builder(op_type, cfg=None):
    A_info = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 4])
    B_info = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.INT8, [4, 3])
    b_scale_info = onnx.helper.make_tensor_value_info("b_scale", onnx.TensorProto.FLOAT, [])
    b_zero_point_info = onnx.helper.make_tensor_value_info("b_zero_point", onnx.TensorProto.INT8, [])

    Y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DynamicQuantizeMatMul",
        inputs=["A", "B", "b_scale", "b_zero_point"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "DynamicQuantizeMatMulGraph",
        [A_info, B_info, b_scale_info, b_zero_point_info],
        [Y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def dynamicquantizematmul_input_generator(session):
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randint(-128, 127, size=(4, 3), dtype=np.int8)
    b_scale = np.array(0.05, dtype=np.float32)
    b_zero_point = np.array(0, dtype=np.int8)

    return {
        "A": A,
        "B": B,
        "b_scale": b_scale,
        "b_zero_point": b_zero_point
    }

SpecialModelBuilders["com.microsoft.DynamicQuantizeMatMul"] = dynamicquantizematmul_model_builder
SpecialInputGenerators["com.microsoft.DynamicQuantizeMatMul"] = dynamicquantizematmul_input_generator
