# ops/microsoft_reducesuminteger.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def reducesuminteger_model_builder(op_type, cfg=None):
    input_shape = [2, 3]
    input_type = onnx.TensorProto.INT8
    output_type = onnx.TensorProto.INT32

    x = onnx.helper.make_tensor_value_info("data", input_type, input_shape)
    y = onnx.helper.make_tensor_value_info("reduced", output_type, [2, 1])

    node = onnx.helper.make_node(
        "ReduceSumInteger",
        inputs=["data"],
        outputs=["reduced"],
        domain="com.microsoft",
        axes=[1],
        keepdims=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "ReduceSumIntegerGraph",
        [x],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def reducesuminteger_input_generator(session):
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return {"data": np.random.randint(-128, 127, size=shape, dtype=np.int8)}

SpecialModelBuilders["com.microsoft.ReduceSumInteger"] = reducesuminteger_model_builder
SpecialInputGenerators["com.microsoft.ReduceSumInteger"] = reducesuminteger_input_generator
