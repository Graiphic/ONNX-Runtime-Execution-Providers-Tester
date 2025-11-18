# ops/matmulinteger.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def matmulinteger_model_builder(op_type, cfg=None):
    a = onnx.helper.make_tensor_value_info("A", TensorProto.UINT8, [2, 3])
    b = onnx.helper.make_tensor_value_info("B", TensorProto.UINT8, [3, 4])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.INT32, None)

    a_zp = onnx.helper.make_tensor("a_zero_point", TensorProto.UINT8, [], [128])
    b_zp = onnx.helper.make_tensor("b_zero_point", TensorProto.UINT8, [], [127])

    node = onnx.helper.make_node(
        "MatMulInteger",
        inputs=["A", "B", "a_zero_point", "b_zero_point"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="matmulinteger_graph",
        inputs=[a, b],
        outputs=[y],
        initializer=[a_zp, b_zp]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def matmulinteger_input_generator(session):
    A = np.random.randint(0, 256, size=(2, 3), dtype=np.uint8)
    B = np.random.randint(0, 256, size=(3, 4), dtype=np.uint8)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["MatMulInteger"] = matmulinteger_model_builder
SpecialInputGenerators["MatMulInteger"] = matmulinteger_input_generator
