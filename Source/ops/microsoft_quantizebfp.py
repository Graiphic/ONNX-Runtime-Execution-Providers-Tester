# ops/microsoft_quantizebfp.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def quantizebfp_model_builder(op_type, cfg=None):
    shape = [2, 3]

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.UINT8, None)
    shape_out = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, None)
    strides_out = onnx.helper.make_tensor_value_info("strides", onnx.TensorProto.INT64, None)

    node = onnx.helper.make_node(
        "QuantizeBFP",
        inputs=["x"],
        outputs=["y", "shape", "strides"],
        domain="com.microsoft",
        bfp_type=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "QuantizeBFPGraph",
        [x],
        [y, shape_out, strides_out]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def quantizebfp_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    return {input_info.name: np.random.rand(*shape).astype(np.float32)}

SpecialModelBuilders["com.microsoft.QuantizeBFP"] = quantizebfp_model_builder
SpecialInputGenerators["com.microsoft.QuantizeBFP"] = quantizebfp_input_generator
