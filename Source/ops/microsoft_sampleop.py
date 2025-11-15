# ops/microsoft_sampleop.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def sampleop_model_builder(op_type, cfg=None):
    dtype = onnx.TensorProto.FLOAT
    input_shape = [2, 3]

    input_tensor = onnx.helper.make_tensor_value_info("X", dtype, input_shape)
    output_tensor = onnx.helper.make_tensor_value_info("Y", dtype, input_shape)

    node = onnx.helper.make_node(
        "SampleOp",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "SampleOpGraph",
        [input_tensor],
        [output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def sampleop_input_generator(session):
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    return {"X": np.random.rand(*shape).astype(np.float32)}

SpecialModelBuilders["com.microsoft.SampleOp"] = sampleop_model_builder
SpecialInputGenerators["com.microsoft.SampleOp"] = sampleop_input_generator
