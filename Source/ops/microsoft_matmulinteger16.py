# ops/microsoft_matmulinteger16.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def matmulinteger16_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.INT16, [2, 3])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.INT16, [3, 4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.INT32, [2, 4])

    node = onnx.helper.make_node(
        "MatMulInteger16",
        inputs=["A", "B"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "MatMulInteger16Graph",
        [A, B],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def matmulinteger16_input_generator(session):
    A = np.random.randint(-100, 100, size=(2, 3), dtype=np.int16)
    B = np.random.randint(-100, 100, size=(3, 4), dtype=np.int16)
    return {"A": A, "B": B}

SpecialModelBuilders["com.microsoft.MatMulInteger16"] = matmulinteger16_model_builder
SpecialInputGenerators["com.microsoft.MatMulInteger16"] = matmulinteger16_input_generator
