# ops/instancenorm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def instancenorm_model_builder(op_type, cfg=None):
    X     = onnx.helper.make_tensor_value_info("X",     onnx.TensorProto.FLOAT, [1, 3, 4, 4])
    scale = onnx.helper.make_tensor_value_info("scale", onnx.TensorProto.FLOAT, [3])
    B     = onnx.helper.make_tensor_value_info("B",     onnx.TensorProto.FLOAT, [3])
    Y     = onnx.helper.make_tensor_value_info("Y",     onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "InstanceNormalization",
        inputs=["X", "scale", "B"],
        outputs=["Y"],
        epsilon=1e-5
    )

    graph = onnx.helper.make_graph([node], "instancenorm_graph", [X, scale, B], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def instancenorm_input_generator(session):
    X = np.random.rand(1, 3, 4, 4).astype(np.float32)
    scale = np.ones(3, dtype=np.float32)
    B     = np.zeros(3, dtype=np.float32)
    return {
        "X": X,
        "scale": scale,
        "B": B
    }

SpecialModelBuilders["InstanceNormalization"] = instancenorm_model_builder
SpecialInputGenerators["InstanceNormalization"] = instancenorm_input_generator
