# ops/microsoft_rfft.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def rfft_model_builder(op_type, cfg=None):
    n = 4
    dtype = onnx.TensorProto.FLOAT

    x = onnx.helper.make_tensor_value_info("X", dtype, [n])
    y = onnx.helper.make_tensor_value_info("Y", dtype, [n // 2 + 1, 2])

    node = onnx.helper.make_node(
        "Rfft",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
        normalized=0,
        onesided=1,
        signal_ndim=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "RfftGraph",
        [x],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def rfft_input_generator(session):
    return {"X": np.random.rand(4).astype(np.float32)}

SpecialModelBuilders["com.microsoft.Rfft"] = rfft_model_builder
SpecialInputGenerators["com.microsoft.Rfft"] = rfft_input_generator
