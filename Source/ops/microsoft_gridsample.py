# ops/gridsample.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gridsample_model_builder(op_type, cfg=None):
    X     = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 4, 4])
    Grid  = onnx.helper.make_tensor_value_info("Grid", onnx.TensorProto.FLOAT, [1, 2, 2, 2])
    Y     = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["X", "Grid"],
        outputs=["Y"],
        domain="com.microsoft",
        mode="bilinear",
        padding_mode="zeros",
        align_corners=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "GridSampleGraph",
        [X, Grid],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def gridsample_input_generator(session):
    X_shape = (1, 3, 4, 4)
    Grid_shape = (1, 2, 2, 2)
    X = np.random.randn(*X_shape).astype(np.float32)
    Grid = np.random.uniform(low=-1, high=1, size=Grid_shape).astype(np.float32)
    return {
        "X": X,
        "Grid": Grid
    }

SpecialModelBuilders["com.microsoft.GridSample"] = gridsample_model_builder
SpecialInputGenerators["com.microsoft.GridSample"] = gridsample_input_generator
