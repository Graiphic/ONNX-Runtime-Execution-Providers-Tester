# ops/microsoft_matmulintegertofloat.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def matmulintegertofloat_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.INT8, [2, 3])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.INT8, [3, 4])
    a_scale = onnx.helper.make_tensor_value_info("a_scale", onnx.TensorProto.FLOAT, [])
    b_scale = onnx.helper.make_tensor_value_info("b_scale", onnx.TensorProto.FLOAT, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])

    node = onnx.helper.make_node(
        "MatMulIntegerToFloat",
        inputs=["A", "B", "a_scale", "b_scale"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "MatMulIntegerToFloatGraph",
        [A, B, a_scale, b_scale],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def matmulintegertofloat_input_generator(session):
    A = np.random.randint(-128, 127, size=(2, 3), dtype=np.int8)
    B = np.random.randint(-128, 127, size=(3, 4), dtype=np.int8)
    a_scale = np.array(0.05, dtype=np.float32)
    b_scale = np.array(0.02, dtype=np.float32)
    return {
        "A": A,
        "B": B,
        "a_scale": a_scale,
        "b_scale": b_scale
    }

SpecialModelBuilders["com.microsoft.MatMulIntegerToFloat"] = matmulintegertofloat_model_builder
SpecialInputGenerators["com.microsoft.MatMulIntegerToFloat"] = matmulintegertofloat_input_generator
