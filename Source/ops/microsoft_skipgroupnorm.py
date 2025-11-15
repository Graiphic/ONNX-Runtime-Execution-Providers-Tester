# ops/microsoft_skipgroupnorm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION


def skipgroupnorm_model_builder(op_type, cfg=None):
    N, H, W, C = 1, 4, 4, 8
    groups = 4
    epsilon = 1e-5
    activation = 0  # None
    channels_last = 1

    def make_input(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    inputs_info = [
        make_input("X", onnx.TensorProto.FLOAT, [N, H, W, C]),
        make_input("gamma", onnx.TensorProto.FLOAT, [C]),
        make_input("beta", onnx.TensorProto.FLOAT, [C]),
        make_input("skip", onnx.TensorProto.FLOAT, [N, H, W, C]),
    ]
    inputs_info.append(make_input("bias", onnx.TensorProto.FLOAT, [C]))

    output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [N, H, W, C])

    node = onnx.helper.make_node(
        "SkipGroupNorm",
        inputs=["X", "gamma", "beta", "skip", "bias"],
        outputs=["Y"],
        domain="com.microsoft",
        activation=activation,
        channels_last=channels_last,
        epsilon=epsilon,
        groups=groups
    )

    graph = onnx.helper.make_graph(
        [node],
        "SkipGroupNormGraph",
        inputs_info,
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model


def skipgroupnorm_input_generator(session):
    inputs = session.get_inputs()
    feed = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed


SpecialModelBuilders["com.microsoft.SkipGroupNorm"] = skipgroupnorm_model_builder
SpecialInputGenerators["com.microsoft.SkipGroupNorm"] = skipgroupnorm_input_generator
