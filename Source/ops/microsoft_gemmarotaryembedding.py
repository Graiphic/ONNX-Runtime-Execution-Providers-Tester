# ops/gemmarotaryembedding.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gemmarotaryembedding_model_builder(op_type, cfg=None):
    emb_info = onnx.helper.make_tensor_value_info("emb", onnx.TensorProto.FLOAT, [2, 4, 64])
    q_info = onnx.helper.make_tensor_value_info("q", onnx.TensorProto.FLOAT16, [2, 8, 4, 64])
    q_rot_info = onnx.helper.make_tensor_value_info("q_rot", onnx.TensorProto.FLOAT16, [2, 8, 4, 64])
    k_info = onnx.helper.make_tensor_value_info("k", onnx.TensorProto.FLOAT16, [2, 8, 4, 64])
    k_rot_info = onnx.helper.make_tensor_value_info("k_rot", onnx.TensorProto.FLOAT16, [2, 8, 4, 64])

    out1_info = onnx.helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT16, None)
    out2_info = onnx.helper.make_tensor_value_info("output2", onnx.TensorProto.FLOAT16, None)

    node = onnx.helper.make_node(
        "GemmaRotaryEmbedding",
        inputs=["emb", "q", "q_rot", "k", "k_rot"],
        outputs=["output1", "output2"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "GemmaRotaryEmbeddingGraph",
        [emb_info, q_info, q_rot_info, k_info, k_rot_info],
        [out1_info, out2_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gemmarotaryembedding_input_generator(session):
    emb = np.random.randn(2, 4, 64).astype(np.float32)
    q = np.random.randn(2, 8, 4, 64).astype(np.float16)
    q_rot = np.random.randn(2, 8, 4, 64).astype(np.float16)
    k = np.random.randn(2, 8, 4, 64).astype(np.float16)
    k_rot = np.random.randn(2, 8, 4, 64).astype(np.float16)

    return {
        "emb": emb,
        "q": q,
        "q_rot": q_rot,
        "k": k,
        "k_rot": k_rot
    }

SpecialModelBuilders["com.microsoft.GemmaRotaryEmbedding"] = gemmarotaryembedding_model_builder
SpecialInputGenerators["com.microsoft.GemmaRotaryEmbedding"] = gemmarotaryembedding_input_generator
