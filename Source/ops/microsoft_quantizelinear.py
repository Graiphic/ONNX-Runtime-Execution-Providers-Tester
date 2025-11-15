# ops/microsoft_quantizelinear.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def quantizelinear_model_builder(op_type, cfg=None):
    shape = [2, 3]

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.UINT8, shape)

    # Initializers (scale, zero_point)
    inits = [
        onnx.helper.make_tensor("y_scale", onnx.TensorProto.FLOAT, [], [0.02]),
        onnx.helper.make_tensor("y_zero_point", onnx.TensorProto.UINT8, [], [128])
    ]

    node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=["x", "y_scale", "y_zero_point"],
        outputs=["y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "QuantizeLinearGraph",
        [x],
        [y],
        initializer=inits
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def quantizelinear_input_generator(session):
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return {"x": np.random.randn(*shape).astype(np.float32)}

SpecialModelBuilders["com.microsoft.QuantizeLinear"] = quantizelinear_model_builder
SpecialInputGenerators["com.microsoft.QuantizeLinear"] = quantizelinear_input_generator
