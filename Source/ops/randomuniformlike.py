# ops/randomuniformlike.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def randomuniformlike_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RandomUniformLike",
        inputs=["X"],
        outputs=["Y"],
        low=1.0,
        high=2.0,
        dtype=TensorProto.FLOAT,
        seed=7.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "randomuniformlike_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def randomuniformlike_input_generator(session):
    return {"X": np.ones((3, 3), dtype=np.float32)}  # shape utilisée seulement

SpecialModelBuilders["RandomUniformLike"] = randomuniformlike_model_builder
SpecialInputGenerators["RandomUniformLike"] = randomuniformlike_input_generator
