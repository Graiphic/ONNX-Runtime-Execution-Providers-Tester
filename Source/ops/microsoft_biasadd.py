# ops/biasadd.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def biasadd_model_builder(op_type, cfg=None):
    shape = [2, 16, 16]  # (N, S, C)
    N, S, C = shape

    # Définition des entrées
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [C])
    skip = onnx.helper.make_tensor_value_info("skip", onnx.TensorProto.FLOAT, shape)

    # Sortie
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)

    # Nœud BiasAdd
    node = onnx.helper.make_node(
        "BiasAdd",
        inputs=["X", "bias", "skip"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "BiasAddGraph",
        [X, bias, skip],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def biasadd_input_generator(session):
    N, S, C = 2, 16, 16
    X = np.random.randn(N, S, C).astype(np.float32)
    bias = np.random.randn(C).astype(np.float32)
    skip = np.random.randn(N, S, C).astype(np.float32)
    return {
        "X": X,
        "bias": bias,
        "skip": skip
    }

SpecialModelBuilders["com.microsoft.BiasAdd"] = biasadd_model_builder
SpecialInputGenerators["com.microsoft.BiasAdd"] = biasadd_input_generator
