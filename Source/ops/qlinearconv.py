# ops/qlinearconv.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def qlinearconv_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 5, 5])
    w = onnx.helper.make_tensor_value_info("w", TensorProto.UINT8, [1, 1, 3, 3])
    y = onnx.helper.make_tensor_value_info("y", TensorProto.UINT8, None)

    # Constantes pour quantification
    x_scale = onnx.helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.1])
    x_zero = onnx.helper.make_tensor("x_zero", TensorProto.UINT8, [], [128])
    w_scale = onnx.helper.make_tensor("w_scale", TensorProto.FLOAT, [], [0.2])
    w_zero = onnx.helper.make_tensor("w_zero", TensorProto.UINT8, [], [127])
    y_scale = onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.3])
    y_zero = onnx.helper.make_tensor("y_zero", TensorProto.UINT8, [], [126])

    node = onnx.helper.make_node(
        "QLinearConv",
        inputs=["x", "x_scale", "x_zero", "w", "w_scale", "w_zero", "y_scale", "y_zero"],
        outputs=["y"],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )

    graph = onnx.helper.make_graph(
        [node],
        "qlinearconv_graph",
        inputs=[x, w],
        outputs=[y],
        initializer=[x_scale, x_zero, w_scale, w_zero, y_scale, y_zero]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def qlinearconv_input_generator(session):
    x = np.random.randint(0, 255, size=(1, 1, 5, 5), dtype=np.uint8)
    w = np.random.randint(0, 255, size=(1, 1, 3, 3), dtype=np.uint8)
    return {"x": x, "w": w}

SpecialModelBuilders["QLinearConv"] = qlinearconv_model_builder
SpecialInputGenerators["QLinearConv"] = qlinearconv_input_generator
