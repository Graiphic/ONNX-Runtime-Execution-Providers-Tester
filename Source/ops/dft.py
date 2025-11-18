# ops/dft.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def dft_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 16, 1])  # réel en [batch, time, 1]
    dft_length = onnx.helper.make_tensor_value_info("dft_length", TensorProto.INT64, [1])
    axis = onnx.helper.make_tensor_value_info("axis", TensorProto.INT64, [1])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DFT",
        inputs=["X", "dft_length", "axis"],
        outputs=["Y"],
        inverse=0,
        onesided=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "dft_graph",
        [x, dft_length, axis],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["DFT"] = dft_model_builder

def dft_input_generator(session):
    x_info, dft_len_info, axis_info = session.get_inputs()
    x = np.random.rand(2, 16, 1).astype(np.float32)
    dft_length = np.array([16], dtype=np.int64)
    axis = np.array([1], dtype=np.int64)
    return {
        x_info.name: x,
        dft_len_info.name: dft_length,
        axis_info.name: axis
    }

SpecialInputGenerators["DFT"] = dft_input_generator
