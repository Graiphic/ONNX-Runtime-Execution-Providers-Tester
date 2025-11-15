# ops/randomuniform.py
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def randomuniform_model_builder(op_type, cfg=None):
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["Y"],
        shape=[2, 4],
        low=0.0,
        high=10.0,
        dtype=TensorProto.FLOAT,
        seed=99.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "randomuniform_graph",
        inputs=[],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def randomuniform_input_generator(session):
    return {}  # aucun input

SpecialModelBuilders["RandomUniform"] = randomuniform_model_builder
SpecialInputGenerators["RandomUniform"] = randomuniform_input_generator
