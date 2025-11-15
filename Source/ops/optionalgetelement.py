# ops/optionalgetelement.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def optionalgetelement_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Crée un Optional puis l'extrait avec GetElement
    to_opt = onnx.helper.make_node("Optional", ["X"], ["opt"])
    get_elem = onnx.helper.make_node("OptionalGetElement", ["opt"], ["Y"])

    graph = onnx.helper.make_graph(
        [to_opt, get_elem],
        "optionalget_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def optionalgetelement_input_generator(session):
    return {"X": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}

SpecialModelBuilders["OptionalGetElement"] = optionalgetelement_model_builder
SpecialInputGenerators["OptionalGetElement"] = optionalgetelement_input_generator
