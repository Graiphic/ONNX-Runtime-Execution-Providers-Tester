# ops/dynamicquantizelinear.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def dynamicquantizelinear_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.UINT8, None)
    y_scale = onnx.helper.make_tensor_value_info("Y_scale", TensorProto.FLOAT, None)
    y_zero_point = onnx.helper.make_tensor_value_info("Y_zero_point", TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "DynamicQuantizeLinear",
        inputs=["X"],
        outputs=["Y", "Y_scale", "Y_zero_point"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "dynamicquantizelinear_graph",
        [x],
        [y, y_scale, y_zero_point]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["DynamicQuantizeLinear"] = dynamicquantizelinear_model_builder

def dynamicquantizelinear_input_generator(session):
    x_info = session.get_inputs()[0]
    shape = [d or 1 for d in x_info.shape]
    x = (np.random.rand(*shape).astype(np.float32) - 0.5) * 2  # valeurs centrées
    return {x_info.name: x}

SpecialInputGenerators["DynamicQuantizeLinear"] = dynamicquantizelinear_input_generator
