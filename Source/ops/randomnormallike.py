# ops/randomnormallike.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def randomnormallike_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RandomNormalLike",
        inputs=["X"],
        outputs=["Y"],
        mean=0.0,
        scale=1.0,
        dtype=TensorProto.FLOAT,
        seed=42.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "randomnormallike_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def randomnormallike_input_generator(session):
    return {"X": np.zeros((4, 3), dtype=np.float32)}  # shape only is used

SpecialModelBuilders["RandomNormalLike"] = randomnormallike_model_builder
SpecialInputGenerators["RandomNormalLike"] = randomnormallike_input_generator
