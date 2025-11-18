# ops/global_pool.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

GLOBAL_POOL_OPS = ["GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool"]

def global_pool_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    attrs = {}
    if op_type == "GlobalLpPool":
        attrs["p"] = 2  # Par défaut p=2 pour L2 pooling

    node = onnx.helper.make_node(op_type, inputs=["X"], outputs=["Y"], **attrs)

    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [X], [Y])
    if op_type == "GlobalLpPool":
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
        )
    else:
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
        )
    
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def global_pool_input_generator(session):
    X = np.random.rand(1, 3, 16, 16).astype(np.float32)
    return {"X": X}

for op in GLOBAL_POOL_OPS:
    SpecialModelBuilders[op] = global_pool_model_builder
    SpecialInputGenerators[op] = global_pool_input_generator
