# ops/microsoft_quickgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def quickgelu_model_builder(op_type, cfg=None):
    shape = [3, 4]

    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, shape)
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        "QuickGelu",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=1.702
    )

    graph = onnx.helper.make_graph(
        [node],
        "QuickGeluGraph",
        [x],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def quickgelu_input_generator(session):
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return {"X": np.random.randn(*shape).astype(np.float32)}

SpecialModelBuilders["com.microsoft.QuickGelu"] = quickgelu_model_builder
SpecialInputGenerators["com.microsoft.QuickGelu"] = quickgelu_input_generator
