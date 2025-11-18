# ops/concat.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def concat_model_builder(op_type, cfg=None):
    # Trois inputs de shape (2, 3)
    inputs = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2, 3])
        for name in ["X0", "X1", "X2"]
    ]
    output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Concat",
        inputs=["X0", "X1", "X2"],
        outputs=["Y"],
        axis=0  # concaténation sur la première dimension
    )

    graph = onnx.helper.make_graph([node], "concat_graph", inputs, [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def concat_input_generator(session):
    inputs = session.get_inputs()
    shape = [d if isinstance(d, int) else 1 for d in inputs[0].shape]
    return {
        inp.name: np.random.rand(*shape).astype(np.float32)
        for inp in inputs
    }

SpecialModelBuilders["Concat"] = concat_model_builder
SpecialInputGenerators["Concat"] = concat_input_generator
