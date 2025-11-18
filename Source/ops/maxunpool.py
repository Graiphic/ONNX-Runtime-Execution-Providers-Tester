# ops/maxunpool.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def maxunpool_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2])
    indices = onnx.helper.make_tensor_value_info("Indices", TensorProto.INT64, [1, 1, 2, 2])
    shape = onnx.helper.make_tensor_value_info("OutputShape", TensorProto.INT64, [4])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "MaxUnpool",
        inputs=["X", "Indices", "OutputShape"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )

    graph = onnx.helper.make_graph(
        [node],
        "maxunpool_graph",
        inputs=[x, indices, shape],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def maxunpool_input_generator(session):
    # Entrée simulée et indices (extrait d’un MaxPool avec return_indices)
    X = np.array([[[[5, 6],
                    [7, 8]]]], dtype=np.float32)
    Indices = np.array([[[[5, 7],
                          [13, 15]]]], dtype=np.int64)
    OutputShape = np.array([1, 1, 4, 4], dtype=np.int64)

    return {
        "X": X,
        "Indices": Indices,
        "OutputShape": OutputShape
    }

SpecialModelBuilders["MaxUnpool"] = maxunpool_model_builder
SpecialInputGenerators["MaxUnpool"] = maxunpool_input_generator
