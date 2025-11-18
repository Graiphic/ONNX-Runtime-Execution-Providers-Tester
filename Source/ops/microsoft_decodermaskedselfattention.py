# ops/decodermaskedmultiheadattention.py

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

# paramètres compatibles CPU / TensorRT / CUDA
BATCH       = 2
SEQ         = 1
NUM_HEADS   = 4
HEAD_SIZE   = 32                  # ← 32 au lieu de 16
HIDDEN_SIZE = NUM_HEADS * HEAD_SIZE  # 128

# ---------- modèle ----------
def decodermaskedmha_model_builder(op_type, cfg=None):
    q = helper.make_tensor_value_info("query", TensorProto.FLOAT, [BATCH, SEQ, HIDDEN_SIZE])
    k = helper.make_tensor_value_info("key",   TensorProto.FLOAT, [BATCH, SEQ, HIDDEN_SIZE])
    v = helper.make_tensor_value_info("value", TensorProto.FLOAT, [BATCH, SEQ, HIDDEN_SIZE])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [BATCH, SEQ, HIDDEN_SIZE])

    node = helper.make_node(
        "DecoderMaskedMultiHeadAttention",
        inputs=["query", "key", "value"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        mask_filter_value=-10000.0
    )

    graph = helper.make_graph([node], "DecMaskedMhaGraph", [q, k, v], [out])

    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

# ---------- générateur d’entrées ----------
def decodermaskedmha_input_generator(session):
    r = np.random.randn
    return {
        "query": r(BATCH, SEQ, HIDDEN_SIZE).astype(np.float32),
        "key":   r(BATCH, SEQ, HIDDEN_SIZE).astype(np.float32),
        "value": r(BATCH, SEQ, HIDDEN_SIZE).astype(np.float32)
    }

SpecialModelBuilders["com.microsoft.DecoderMaskedMultiHeadAttention"]   = decodermaskedmha_model_builder
SpecialInputGenerators["com.microsoft.DecoderMaskedMultiHeadAttention"] = decodermaskedmha_input_generator
