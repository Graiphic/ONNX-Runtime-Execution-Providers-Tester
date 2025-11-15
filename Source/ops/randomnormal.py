# ops/randomnormal.py
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def randomnormal_model_builder(op_type, cfg=None):
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RandomNormal",
        inputs=[],
        outputs=["Y"],
        shape=[3, 2],
        dtype=TensorProto.FLOAT,
        mean=0.0,
        scale=1.0,
        seed=123.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "randomnormal_graph",
        inputs=[],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def randomnormal_input_generator(session):
    return {}  # aucun input requis

SpecialModelBuilders["RandomNormal"] = randomnormal_model_builder
SpecialInputGenerators["RandomNormal"] = randomnormal_input_generator
