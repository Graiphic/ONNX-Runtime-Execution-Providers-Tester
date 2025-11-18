# ops/expanddims.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def expanddims_model_builder(op_type, cfg=None):
    input_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
    axis_info = onnx.helper.make_tensor_value_info("axis", onnx.TensorProto.INT32, [1])

    output_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "ExpandDims",
        inputs=["X", "axis"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "ExpandDimsGraph",
        [input_info, axis_info],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def expanddims_input_generator(session):
    X = np.random.randn(2, 3).astype(np.float32)
    axis = np.array([1], dtype=np.int32)  # insère une nouvelle dimension à l'indice 1
    return {
        "X": X,
        "axis": axis
    }

SpecialModelBuilders["com.microsoft.ExpandDims"] = expanddims_model_builder
SpecialInputGenerators["com.microsoft.ExpandDims"] = expanddims_input_generator
