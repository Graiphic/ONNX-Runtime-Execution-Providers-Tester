# ops/resize.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def resize_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    roi = onnx.helper.make_tensor_value_info("roi", onnx.TensorProto.FLOAT, [0])  # Entrée facultative, vide ici
    scales = onnx.helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [0])  # Entrée facultative, vide ici
    sizes = onnx.helper.make_tensor_value_info("sizes", onnx.TensorProto.INT64, [4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Resize",
        inputs=["X", "", "", "sizes"],
        outputs=["Y"],
        mode="linear",
        coordinate_transformation_mode="align_corners",
        nearest_mode="round_prefer_floor"
    )

    graph = onnx.helper.make_graph([node], "resize_graph", [X, roi, scales, sizes], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def resize_input_generator(session):
    X = np.random.rand(1, 3, 16, 16).astype(np.float32)
    roi = np.array([], dtype=np.float32)
    scales = np.array([], dtype=np.float32)
    sizes = np.array([1, 3, 32, 32], dtype=np.int64)  # Double la taille HxW
    return {
        "X": X,
        "roi": roi,
        "scales": scales,
        "sizes": sizes
    }

SpecialModelBuilders["Resize"] = resize_model_builder
SpecialInputGenerators["Resize"] = resize_input_generator
