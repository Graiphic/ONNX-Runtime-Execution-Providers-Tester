# ops/gridsample.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gridsample_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 16, 16])
    grid = onnx.helper.make_tensor_value_info("grid", onnx.TensorProto.FLOAT, [1, 16, 16, 2])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="linear",
        padding_mode="zeros",
        align_corners=1
    )

    graph = onnx.helper.make_graph([node], "gridsample_graph", [X, grid], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def gridsample_input_generator(session):
    X = np.random.rand(1, 1, 16, 16).astype(np.float32)
    # Génère une grille d'identité normalisée
    lin = np.linspace(-1, 1, 16)
    grid_x, grid_y = np.meshgrid(lin, lin)
    grid = np.stack((grid_x, grid_y), axis=-1)
    grid = np.expand_dims(grid, axis=0).astype(np.float32)
    return {
        "X": X,
        "grid": grid
    }

SpecialModelBuilders["GridSample"] = gridsample_model_builder
SpecialInputGenerators["GridSample"] = gridsample_input_generator
