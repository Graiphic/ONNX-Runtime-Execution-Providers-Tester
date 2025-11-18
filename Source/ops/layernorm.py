# ops/layernorm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def layernorm_model_builder(op_type, cfg=None):
    X     = onnx.helper.make_tensor_value_info("X",     onnx.TensorProto.FLOAT, [2, 4, 8])
    scale = onnx.helper.make_tensor_value_info("scale", onnx.TensorProto.FLOAT, [8])
    B     = onnx.helper.make_tensor_value_info("B",     onnx.TensorProto.FLOAT, [8])
    Y     = onnx.helper.make_tensor_value_info("Y",     onnx.TensorProto.FLOAT, None)
    Mean     = onnx.helper.make_tensor_value_info("Mean",     onnx.TensorProto.FLOAT, None)
    InvStdDev     = onnx.helper.make_tensor_value_info("InvStdDev",     onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "LayerNormalization",
        inputs=["X", "scale", "B"],
        outputs=["Y", "Mean", "InvStdDev"],
        axis=-1,
        epsilon=1e-5,
        stash_type=1
    )

    graph = onnx.helper.make_graph([node], "layernorm_graph", [X, scale, B], [Y, Mean, InvStdDev])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def layernorm_input_generator(session):
    X     = np.random.rand(2, 4, 8).astype(np.float32)
    scale = np.ones(8, dtype=np.float32)
    B     = np.zeros(8, dtype=np.float32)
    return {
        "X": X,
        "scale": scale,
        "B": B
    }

SpecialModelBuilders["LayerNormalization"] = layernorm_model_builder
SpecialInputGenerators["LayerNormalization"] = layernorm_input_generator
