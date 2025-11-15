# ops/multinomial.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def multinomial_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.INT32, None)

    node = onnx.helper.make_node(
        "Multinomial",
        inputs=["X"],
        outputs=["Y"],
        sample_size=2,
        dtype=TensorProto.INT32,
        seed=42.0
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="multinomial_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def multinomial_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    x /= np.sum(x, axis=1, keepdims=True)  # normalisation ligne par ligne
    return {input_info.name: x}

SpecialModelBuilders["Multinomial"] = multinomial_model_builder
SpecialInputGenerators["Multinomial"] = multinomial_input_generator
