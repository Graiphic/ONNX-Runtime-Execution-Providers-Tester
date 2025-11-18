# ops/lrn.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def lrn_model_builder(op_type, cfg=None):
    inp = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 5, 5])
    out = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "LRN",
        inputs=["X"],
        outputs=["Y"],
        size=3,
        alpha=0.0001,
        beta=0.75,
        bias=1.0
    )

    graph = onnx.helper.make_graph([node], "lrn_graph", [inp], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def lrn_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {input_info.name: x}

SpecialModelBuilders["LRN"] = lrn_model_builder
SpecialInputGenerators["LRN"] = lrn_input_generator
