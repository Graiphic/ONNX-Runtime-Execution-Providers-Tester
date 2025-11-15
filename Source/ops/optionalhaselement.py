# ops/optionalhaselement.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def optionalhaselement_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    has = onnx.helper.make_tensor_value_info("H", TensorProto.BOOL, None)

    to_opt = onnx.helper.make_node("Optional", ["X"], ["opt"])
    check = onnx.helper.make_node("OptionalHasElement", ["opt"], ["H"])

    graph = onnx.helper.make_graph(
        [to_opt, check],
        "optionalhas_graph",
        inputs=[x],
        outputs=[has]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def optionalhaselement_input_generator(session):
    return {"X": np.array([0.5, 1.5], dtype=np.float32)}

SpecialModelBuilders["OptionalHasElement"] = optionalhaselement_model_builder
SpecialInputGenerators["OptionalHasElement"] = optionalhaselement_input_generator
