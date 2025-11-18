# ops/embedlayernormalization.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def embedlayernormalization_model_builder(op_type, cfg=None):
    batch_size = 2
    seq_len = 4
    vocab_size = 32
    hidden_size = 8

    input_ids_info = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [batch_size, seq_len])
    word_embedding_info = onnx.helper.make_tensor_value_info("word_embedding", onnx.TensorProto.FLOAT, [vocab_size, hidden_size])
    position_embedding_info = onnx.helper.make_tensor_value_info("position_embedding", onnx.TensorProto.FLOAT, [seq_len, hidden_size])
    gamma_info = onnx.helper.make_tensor_value_info("gamma", onnx.TensorProto.FLOAT, [hidden_size])
    beta_info = onnx.helper.make_tensor_value_info("beta", onnx.TensorProto.FLOAT, [hidden_size])

    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)
    mask_index_info = onnx.helper.make_tensor_value_info("mask_index", onnx.TensorProto.INT32, None)
    embedding_sum_info = onnx.helper.make_tensor_value_info("embedding_sum", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "EmbedLayerNormalization",
        inputs=["input_ids", "", "word_embedding", "position_embedding", "", "gamma", "beta"],
        outputs=["output", "mask_index", "embedding_sum"],
        domain="com.microsoft",
        epsilon=1e-5,
        mask_index_type=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "EmbedLayerNormalizationGraph",
        [input_ids_info, word_embedding_info, position_embedding_info, gamma_info, beta_info],
        [output_info, mask_index_info, embedding_sum_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def embedlayernormalization_input_generator(session):
    batch_size = 2
    seq_len = 4
    vocab_size = 32
    hidden_size = 8

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    word_embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    position_embedding = np.random.randn(seq_len, hidden_size).astype(np.float32)
    gamma = np.ones(hidden_size, dtype=np.float32)
    beta = np.zeros(hidden_size, dtype=np.float32)

    return {
        "input_ids": input_ids,
        "word_embedding": word_embedding,
        "position_embedding": position_embedding,
        "gamma": gamma,
        "beta": beta
    }

SpecialModelBuilders["com.microsoft.EmbedLayerNormalization"] = embedlayernormalization_model_builder
SpecialInputGenerators["com.microsoft.EmbedLayerNormalization"] = embedlayernormalization_input_generator
