# ops/nonzero.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def nonzero_model_builder(op_type, cfg=None):
    inp = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
    out = onnx.helper.make_tensor_value_info("Y", TensorProto.INT64, None)

    node = onnx.helper.make_node(
        "NonZero",
        inputs=["X"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "nonzero_graph",
        inputs=[inp],
        outputs=[out]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def nonzero_input_generator(session):
    data = np.random.rand(3, 4).astype(np.float32)
    data[data < 0.3] = 0.0  # force certains zéros
    return {"X": data}

SpecialModelBuilders["NonZero"] = nonzero_model_builder
SpecialInputGenerators["NonZero"] = nonzero_input_generator
