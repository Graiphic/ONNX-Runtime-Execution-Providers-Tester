# ops/microsoft_wordconvembedding.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def wordconvembedding_model_builder(op_type, cfg=None):
    # Shape: [batch_size, seq_len, word_len]
    Sequence = helper.make_tensor_value_info("Sequence", TensorProto.INT32, [1, 2, 5])

    # Conv weights: [num_filters, embedding_size, window_size]
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 8, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4])
    
    # Char embedding: [vocab_size, char_embedding_size]
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [26, 8])

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = helper.make_node(
        "WordConvEmbedding",
        inputs=["Sequence", "W", "B", "C"],
        outputs=["Y"],
        domain="com.microsoft",
        char_embedding_size=8,
        conv_window_size=3,
        embedding_size=4
    )

    graph = helper.make_graph(
        [node],
        "WordConvEmbeddingGraph",
        [Sequence, W, B, C],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def wordconvembedding_input_generator(session):
    # Batch of 1, 2 words, 5 chars per word (values ∈ [0, 25])
    Sequence = np.random.randint(0, 26, size=(1, 2, 5), dtype=np.int32)

    # Conv weights and bias
    W = np.random.randn(4, 8, 3).astype(np.float32)
    B = np.random.randn(4).astype(np.float32)

    # Char embedding: 26 chars, each 8-dim
    C = np.random.randn(26, 8).astype(np.float32)

    return {
        "Sequence": Sequence,
        "W": W,
        "B": B,
        "C": C
    }

SpecialModelBuilders["com.microsoft.WordConvEmbedding"] = wordconvembedding_model_builder
SpecialInputGenerators["com.microsoft.WordConvEmbedding"] = wordconvembedding_input_generator
