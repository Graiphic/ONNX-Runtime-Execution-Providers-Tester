# ops/qlinearmatmul.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def qlinearmatmul_model_builder(op_type, cfg=None):
    a = onnx.helper.make_tensor_value_info("a", TensorProto.UINT8, [2, 3])
    b = onnx.helper.make_tensor_value_info("b", TensorProto.UINT8, [3, 4])
    y = onnx.helper.make_tensor_value_info("y", TensorProto.UINT8, None)

    a_scale = onnx.helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.05])
    a_zero = onnx.helper.make_tensor("a_zero", TensorProto.UINT8, [], [128])
    b_scale = onnx.helper.make_tensor("b_scale", TensorProto.FLOAT, [], [0.04])
    b_zero = onnx.helper.make_tensor("b_zero", TensorProto.UINT8, [], [127])
    y_scale = onnx.helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.03])
    y_zero = onnx.helper.make_tensor("y_zero", TensorProto.UINT8, [], [126])

    node = onnx.helper.make_node(
        "QLinearMatMul",
        inputs=["a", "a_scale", "a_zero", "b", "b_scale", "b_zero", "y_scale", "y_zero"],
        outputs=["y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "qlinearmatmul_graph",
        inputs=[a, b],
        outputs=[y],
        initializer=[a_scale, a_zero, b_scale, b_zero, y_scale, y_zero]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def qlinearmatmul_input_generator(session):
    a = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, size=(3, 4), dtype=np.uint8)
    return {"a": a, "b": b}

SpecialModelBuilders["QLinearMatMul"] = qlinearmatmul_model_builder
SpecialInputGenerators["QLinearMatMul"] = qlinearmatmul_input_generator
