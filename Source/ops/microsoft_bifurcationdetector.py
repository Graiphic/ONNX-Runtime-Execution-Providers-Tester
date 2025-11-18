# ops/bifurcationdetector.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def bifurcationdetector_model_builder(op_type, cfg=None):
    # On définit les tenseurs comme 1-D plutôt que 2-D, 
    # car l'implémentation interne de BifurcationDetector 
    # interprète len(src_tokens) sur la première dimension.
    src_tokens = onnx.helper.make_tensor_value_info(
        "src_tokens", TensorProto.INT64, [None]
    )
    cur_tokens = onnx.helper.make_tensor_value_info(
        "cur_tokens", TensorProto.INT64, [None]
    )
    # prev_suffix_match_idx est un scalaire (shape [])
    prev_idx = onnx.helper.make_tensor_value_info(
        "prev_suffix_match_idx", TensorProto.INT64, []
    )
    pred_tokens = onnx.helper.make_tensor_value_info(
        "pred_tokens", TensorProto.INT64, [None]
    )

    tokens_out = onnx.helper.make_tensor_value_info(
        "tokens", TensorProto.INT64, [None]
    )
    # suffix_match_idx est également un scalaire
    suffix_match_idx = onnx.helper.make_tensor_value_info(
        "suffix_match_idx", TensorProto.INT64, []
    )

    node = onnx.helper.make_node(
        "BifurcationDetector",
        inputs=["src_tokens", "cur_tokens", "prev_suffix_match_idx", "pred_tokens"],
        outputs=["tokens", "suffix_match_idx"],
        domain="com.microsoft",
        min_ngram_size=2,
        max_ngram_size=5
    )

    graph = onnx.helper.make_graph(
        [node],
        "BifurcationDetectorGraph",
        [src_tokens, cur_tokens, prev_idx, pred_tokens],
        [tokens_out, suffix_match_idx]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def bifurcationdetector_input_generator(session):
    src_len = 16
    # On choisit un indice de correspondance entre 0 et src_len - 1
    match_idx = np.random.randint(0, src_len)  # entre 0 et 15

    # On génère src et cur comme 1-D, longueur src_len
    src = np.random.randint(0, 10000, size=(src_len,), dtype=np.int64)
    cur = np.random.randint(0, 10000, size=(src_len,), dtype=np.int64)
    # prev_suffix_match_idx est un scalaire
    idx = np.array(match_idx, dtype=np.int64)

    pred_len = src_len + 1 - match_idx
    pred = np.random.randint(0, 10000, size=(pred_len,), dtype=np.int64)

    return {
        "src_tokens": src,
        "cur_tokens": cur,
        "prev_suffix_match_idx": idx,
        "pred_tokens": pred
    }

SpecialModelBuilders["com.microsoft.BifurcationDetector"] = bifurcationdetector_model_builder
SpecialInputGenerators["com.microsoft.BifurcationDetector"] = bifurcationdetector_input_generator
