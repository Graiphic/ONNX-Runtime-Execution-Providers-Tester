# ops/lpnormalization.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def lpnormalization_model_builder(op_type, cfg=None):
    inp = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    out = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "LpNormalization",
        inputs=["X"],
        outputs=["Y"],
        axis=1,
        p=2  # L2 normalization
    )

    graph = onnx.helper.make_graph([node], "lpnormalization_graph", [inp], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def lpnormalization_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {input_info.name: x}

SpecialModelBuilders["LpNormalization"] = lpnormalization_model_builder
SpecialInputGenerators["LpNormalization"] = lpnormalization_input_generator
