# ops/matmul.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def matmul_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 4])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [4, 3])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"])

    graph = onnx.helper.make_graph([node], "matmul_graph", [A, B], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def matmul_input_generator(session):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 3).astype(np.float32)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["MatMul"] = matmul_model_builder
SpecialInputGenerators["MatMul"] = matmul_input_generator
