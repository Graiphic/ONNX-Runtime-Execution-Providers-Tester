# ops/biassoftmax.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def biassoftmax_model_builder(op_type, cfg=None):
    shape = [2, 12, 64]  # exemple typique (batch, heads, tokens)
    N, H, T = shape

    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, shape)
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [N, H, T])  # même shape ici pour simplicité
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        "BiasSoftmax",
        inputs=["data", "bias"],
        outputs=["output"],
        domain="com.microsoft",
        axis=-1,  # softmax sur dernière dim
        is_inner_broadcast=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "BiasSoftmaxGraph",
        [data, bias],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def biassoftmax_input_generator(session):
    shape = [2, 12, 64]
    data = np.random.randn(*shape).astype(np.float32)
    bias = np.random.randn(*shape).astype(np.float32)
    return {
        "data": data,
        "bias": bias
    }

SpecialModelBuilders["com.microsoft.BiasSoftmax"] = biassoftmax_model_builder
SpecialInputGenerators["com.microsoft.BiasSoftmax"] = biassoftmax_input_generator
