# ops/microsoft_torchembedding.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def torchembedding_model_builder(op_type, cfg=None):
    weight = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [10, 4])  # 10 tokens, 4-dim embeddings
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [3])    # 3 indices to lookup
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)               # output [3, 4]

    node = helper.make_node(
        "TorchEmbedding",
        inputs=["weight", "indices"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = helper.make_graph(
        [node],
        "TorchEmbeddingGraph",
        [weight, indices],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def torchembedding_input_generator(session):
    weight_val = np.random.randn(10, 4).astype(np.float32)
    indices_val = np.random.randint(0, 10, size=(3,), dtype=np.int64)

    return {
        "weight": weight_val,
        "indices": indices_val
    }

SpecialModelBuilders["com.microsoft.TorchEmbedding"] = torchembedding_model_builder
SpecialInputGenerators["com.microsoft.TorchEmbedding"] = torchembedding_input_generator
