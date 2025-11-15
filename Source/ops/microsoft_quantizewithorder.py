# ops/microsoft_quantizewithorder.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def quantizewithorder_model_builder(op_type, cfg=None):
    shape = [4, 8]

    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape)
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, shape)

    inits = [
        onnx.helper.make_tensor("scale_input", onnx.TensorProto.FLOAT, [], [0.05])
    ]

    node = onnx.helper.make_node(
        "QuantizeWithOrder",
        inputs=["input", "scale_input"],
        outputs=["output"],
        domain="com.microsoft",
        order_input=1,
        order_output=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "QuantizeWithOrderGraph",
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

def quantizewithorder_input_generator(session):
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return {"input": np.random.randn(*shape).astype(np.float32)}

SpecialModelBuilders["com.microsoft.QuantizeWithOrder"] = quantizewithorder_model_builder
SpecialInputGenerators["com.microsoft.QuantizeWithOrder"] = quantizewithorder_input_generator
