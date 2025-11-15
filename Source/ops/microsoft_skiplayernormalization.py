# ops/microsoft_skiplayernormalization.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION


def skiplayernorm_model_builder(op_type, cfg=None):
    B, S, H = 2, 4, 8
    epsilon = 1e-5

    def make_input(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    inputs_info = [
        make_input("input", onnx.TensorProto.FLOAT, [B, S, H]),
        make_input("skip", onnx.TensorProto.FLOAT, [B, S, H]),
        make_input("gamma", onnx.TensorProto.FLOAT, [H]),
        make_input("beta", onnx.TensorProto.FLOAT, [H]),
        make_input("bias", onnx.TensorProto.FLOAT, [H])
    ]

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [B, S, H])

    node = onnx.helper.make_node(
        "SkipLayerNormalization",
        inputs=["input", "skip", "gamma", "beta", "bias"],
        outputs=["output"],
        domain="com.microsoft",
        epsilon=epsilon
    )

    graph = onnx.helper.make_graph(
        [node],
        "SkipLayerNormalizationGraph",
        inputs_info,
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model


def skiplayernorm_input_generator(session):
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed


SpecialModelBuilders["com.microsoft.SkipLayerNormalization"] = skiplayernorm_model_builder
SpecialInputGenerators["com.microsoft.SkipLayerNormalization"] = skiplayernorm_input_generator
