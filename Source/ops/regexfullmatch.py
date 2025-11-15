# ops/regexfullmatch.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def regexfullmatch_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.STRING, [None])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.BOOL, None)

    node = onnx.helper.make_node(
        "RegexFullMatch",
        inputs=["X"],
        outputs=["Y"],
        pattern="^a.*z$"  # Correspond aux chaînes commençant par 'a' et se terminant par 'z'
    )

    graph = onnx.helper.make_graph(
        [node],
        "regexfullmatch_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", 20)]  # Version minimale requise
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def regexfullmatch_input_generator(session):
    return {"X": np.array(["abcz", "a123z", "xyz", "az", "a-z"], dtype=object)}

SpecialModelBuilders["RegexFullMatch"] = regexfullmatch_model_builder
SpecialInputGenerators["RegexFullMatch"] = regexfullmatch_input_generator
