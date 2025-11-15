# ops/microsoft_groupnorm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def groupnorm_model_builder(op_type, cfg=None):
    N, H, W, C = 2, 4, 4, 8  # C divisible par groups
    X     = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [N, H, W, C])
    gamma = onnx.helper.make_tensor_value_info("gamma", onnx.TensorProto.FLOAT, [C])
    beta  = onnx.helper.make_tensor_value_info("beta", onnx.TensorProto.FLOAT, [C])
    Y     = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GroupNorm",
        inputs=["X", "gamma", "beta"],
        outputs=["Y"],
        domain="com.microsoft",
        activation=0,
        channels_last=1,  # NHWC layout
        epsilon=1e-5,
        groups=4
    )

    graph = onnx.helper.make_graph(
        [node],
        "GroupNormGraph",
        [X, gamma, beta],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def groupnorm_input_generator(session):
    X_shape = (2, 4, 4, 8)
    C = X_shape[-1]
    X = np.random.randn(*X_shape).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)
    return {
        "X": X,
        "gamma": gamma,
        "beta": beta
    }

SpecialModelBuilders["com.microsoft.GroupNorm"] = groupnorm_model_builder
SpecialInputGenerators["com.microsoft.GroupNorm"] = groupnorm_input_generator
