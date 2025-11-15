# ops/onehot.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def onehot_model_builder(op_type, cfg=None):
    indices = onnx.helper.make_tensor_value_info("indices", TensorProto.INT64, [3])
    depth = onnx.helper.make_tensor("depth", TensorProto.INT64, [], [5])
    values = onnx.helper.make_tensor("values", TensorProto.FLOAT, [2], [0.0, 1.0])
    output = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "OneHot",
        inputs=["indices", "depth", "values"],
        outputs=["Y"],
        axis=-1
    )

    graph = onnx.helper.make_graph(
        [node],
        "onehot_graph",
        inputs=[indices],
        outputs=[output],
        initializer=[depth, values]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def onehot_input_generator(session):
    return {"indices": np.array([0, 2, 4], dtype=np.int64)}

SpecialModelBuilders["OneHot"] = onehot_model_builder
SpecialInputGenerators["OneHot"] = onehot_input_generator
