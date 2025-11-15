# ops/microsoft_inverse.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def inverse_model_builder(op_type, cfg=None):
    # Batch de 2 matrices 3x3
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3, 3])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3, 3])

    node = onnx.helper.make_node(
        "Inverse",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "InverseGraph",
        [X],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def inverse_input_generator(session):
    # Génère des matrices 3x3 bien conditionnées (matrices aléatoires + identité)
    X = np.array([np.eye(3) + 0.1 * np.random.randn(3, 3) for _ in range(2)], dtype=np.float32)
    return {"X": X}

SpecialModelBuilders["com.microsoft.Inverse"] = inverse_model_builder
SpecialInputGenerators["com.microsoft.Inverse"] = inverse_input_generator
