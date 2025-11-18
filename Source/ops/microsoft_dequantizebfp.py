# ops/dequantizebfp.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dequantizebfp_model_builder(op_type, cfg=None):
    # Entrées
    x_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.UINT8, [None])
    shape_info = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [None])
    strides_info = onnx.helper.make_tensor_value_info("strides", onnx.TensorProto.INT64, [None])

    # Sortie
    y_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DequantizeBFP",
        inputs=["x", "shape", "strides"],
        outputs=["y"],
        domain="com.microsoft",
        bfp_type=1,           # Exemple: IEEE BFP type ID = 1 (selon enum BFPType)
        block_dim=-1,         # Par défaut dernière dimension
        dtype=1               # FLOAT (1 = float32)
    )

    graph = onnx.helper.make_graph(
        [node],
        "DequantizeBFPGraph",
        [x_info, shape_info, strides_info],
        [y_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def dequantizebfp_input_generator(session):
    shape = np.array([2, 4], dtype=np.int64)
    strides = np.array([4, 1], dtype=np.int64)  # standard row-major layout
    size = np.prod(shape)
    x = np.random.randint(0, 256, size=size, dtype=np.uint8)
    return {
        "x": x,
        "shape": shape,
        "strides": strides
    }

SpecialModelBuilders["com.microsoft.DequantizeBFP"] = dequantizebfp_model_builder
SpecialInputGenerators["com.microsoft.DequantizeBFP"] = dequantizebfp_input_generator
