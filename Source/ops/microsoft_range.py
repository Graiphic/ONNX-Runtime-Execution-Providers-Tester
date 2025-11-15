# ops/microsoft_range.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def range_model_builder(op_type, cfg=None):
    dtype = onnx.TensorProto.INT32

    # Inputs explicites
    start = onnx.helper.make_tensor_value_info("start", dtype, [])
    limit = onnx.helper.make_tensor_value_info("limit", dtype, [])
    delta = onnx.helper.make_tensor_value_info("delta", dtype, [])

    y = onnx.helper.make_tensor_value_info("Y", dtype, None)

    node = onnx.helper.make_node(
        "Range",
        inputs=["start", "limit", "delta"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "RangeGraph",
        [start, limit, delta],
        [y],
        initializer=[]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def range_input_generator(session):
    # Valeurs identiques à celles initialement dans le modèle
    return {
        "start": np.array(2, dtype=np.int32),
        "limit": np.array(10, dtype=np.int32),
        "delta": np.array(2, dtype=np.int32)
    }

SpecialModelBuilders["com.microsoft.Range"] = range_model_builder
SpecialInputGenerators["com.microsoft.Range"] = range_input_generator
