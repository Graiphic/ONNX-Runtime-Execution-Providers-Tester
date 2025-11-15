# ops/microsoft_qlinearreducemean.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearreducemean_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.UINT8, [2, 4])
    data_scale = onnx.helper.make_tensor_value_info("data_scale", onnx.TensorProto.FLOAT, [])
    data_zp = onnx.helper.make_tensor_value_info("data_zero_point", onnx.TensorProto.UINT8, [])
    reduced_scale = onnx.helper.make_tensor_value_info("reduced_scale", onnx.TensorProto.FLOAT, [])
    reduced_zp = onnx.helper.make_tensor_value_info("reduced_zero_point", onnx.TensorProto.UINT8, [])
    reduced = onnx.helper.make_tensor_value_info("reduced", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearReduceMean",
        inputs=["data", "data_scale", "data_zero_point", "reduced_scale", "reduced_zero_point"],
        outputs=["reduced"],
        domain="com.microsoft",
        axes=[1],
        keepdims=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearReduceMeanGraph",
        [data, data_scale, data_zp, reduced_scale, reduced_zp],
        [reduced]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearreducemean_input_generator(session):
    return {
        "data": np.random.randint(0, 256, size=(2, 4), dtype=np.uint8),
        "data_scale": np.array(0.02, dtype=np.float32),
        "data_zero_point": np.array(128, dtype=np.uint8),
        "reduced_scale": np.array(0.01, dtype=np.float32),
        "reduced_zero_point": np.array(128, dtype=np.uint8)
    }

SpecialModelBuilders["com.microsoft.QLinearReduceMean"] = qlinearreducemean_model_builder
SpecialInputGenerators["com.microsoft.QLinearReduceMean"] = qlinearreducemean_input_generator
