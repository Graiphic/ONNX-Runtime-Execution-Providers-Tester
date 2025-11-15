# ops/quantizelinear.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def quantizelinear_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = onnx.helper.make_tensor_value_info("y", TensorProto.UINT8, None)

    scale = onnx.helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
    zero = onnx.helper.make_tensor("zero", TensorProto.UINT8, [], [128])

    node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=["x", "scale", "zero"],
        outputs=["y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "quantizelinear_graph",
        inputs=[x],
        outputs=[y],
        initializer=[scale, zero]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def quantizelinear_input_generator(session):
    return {"x": np.random.uniform(low=-1.0, high=1.0, size=(2, 2)).astype(np.float32)}

SpecialModelBuilders["QuantizeLinear"] = quantizelinear_model_builder
SpecialInputGenerators["QuantizeLinear"] = quantizelinear_input_generator
