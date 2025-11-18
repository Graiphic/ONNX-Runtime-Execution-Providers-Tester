# ops/biasgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def biasgelu_model_builder(op_type, cfg=None):
    shape = [2, 128]  # shape of A
    bias_shape = [128]  # 1D bias

    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, shape)
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, bias_shape)
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        "BiasGelu",
        inputs=["A", "B"],
        outputs=["C"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "BiasGeluGraph",
        [A, B],
        [C]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def biasgelu_input_generator(session):
    shape = [2, 128]
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(shape[1]).astype(np.float32)
    return {
        "A": A,
        "B": B
    }

SpecialModelBuilders["com.microsoft.BiasGelu"] = biasgelu_model_builder
SpecialInputGenerators["com.microsoft.BiasGelu"] = biasgelu_input_generator
