# ops/decodermaskedmultiheadattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def decodermaskedmha_model_builder(op_type, cfg=None):
    batch_size = 2
    sequence_length = 1
    num_heads = 4
    head_size = 16
    hidden_size = num_heads * head_size

    query = onnx.helper.make_tensor_value_info("query", onnx.TensorProto.FLOAT, [batch_size, sequence_length, hidden_size])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)  # shape infer

    node = onnx.helper.make_node(
        "DecoderMaskedMultiHeadAttention",
        inputs=["query"],  # minimal: only query provided
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        mask_filter_value=-10000.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "DecoderMaskedMhaGraph",
        [query],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def decodermaskedmha_input_generator(session):
    batch_size = 2
    sequence_length = 1
    num_heads = 4
    head_size = 16
    hidden_size = num_heads * head_size

    query = np.random.randn(batch_size, sequence_length, hidden_size).astype(np.float32)
    return {
        "query": query
    }

SpecialModelBuilders["com.microsoft.DecoderMaskedMultiHeadAttention"] = decodermaskedmha_model_builder
SpecialInputGenerators["com.microsoft.DecoderMaskedMultiHeadAttention"] = decodermaskedmha_input_generator
