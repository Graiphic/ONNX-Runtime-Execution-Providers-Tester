# ops/microsoft_murmurhash3.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def murmurhash3_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.STRING, [3])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT32, [3])

    node = onnx.helper.make_node(
        "MurmurHash3",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
        positive=1,
        seed=42
    )

    graph = onnx.helper.make_graph(
        [node],
        "MurmurHash3Graph",
        [X],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def murmurhash3_input_generator(session):
    X = np.array(["apple", "banana", "cherry"], dtype=object)
    return {"X": X}

SpecialModelBuilders["com.microsoft.MurmurHash3"] = murmurhash3_model_builder
SpecialInputGenerators["com.microsoft.MurmurHash3"] = murmurhash3_input_generator
