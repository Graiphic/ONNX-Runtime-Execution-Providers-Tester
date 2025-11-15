# ops/microsoft_qlinearconcat.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearconcat_model_builder(op_type, cfg=None):
    axis = 1

    # Inputs : (tensor, scale, zero_point) for each
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [1, 2])
    A_scale = onnx.helper.make_tensor_value_info("A_scale", onnx.TensorProto.FLOAT, [])
    A_zp = onnx.helper.make_tensor_value_info("A_zero_point", onnx.TensorProto.UINT8, [])

    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [1, 3])
    B_scale = onnx.helper.make_tensor_value_info("B_scale", onnx.TensorProto.FLOAT, [])
    B_zp = onnx.helper.make_tensor_value_info("B_zero_point", onnx.TensorProto.UINT8, [])

    Y_scale = onnx.helper.make_tensor_value_info("Y_scale", onnx.TensorProto.FLOAT, [])
    Y_zp = onnx.helper.make_tensor_value_info("Y_zero_point", onnx.TensorProto.UINT8, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearConcat",
        inputs=["Y_scale", "Y_zero_point", "A", "A_scale", "A_zero_point", "B", "B_scale", "B_zero_point"],
        outputs=["Y"],
        domain="com.microsoft",
        axis=axis
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearConcatGraph",
        [Y_scale, Y_zp, A, A_scale, A_zp, B, B_scale, B_zp],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearconcat_input_generator(session):
    inputs = {
        "A": np.random.randint(0, 256, size=(1, 2), dtype=np.uint8),
        "A_scale": np.array(0.02, dtype=np.float32),
        "A_zero_point": np.array(128, dtype=np.uint8),
        "B": np.random.randint(0, 256, size=(1, 3), dtype=np.uint8),
        "B_scale": np.array(0.02, dtype=np.float32),
        "B_zero_point": np.array(128, dtype=np.uint8),
        "Y_scale": np.array(0.02, dtype=np.float32),
        "Y_zero_point": np.array(128, dtype=np.uint8)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QLinearConcat"] = qlinearconcat_model_builder
SpecialInputGenerators["com.microsoft.QLinearConcat"] = qlinearconcat_input_generator
