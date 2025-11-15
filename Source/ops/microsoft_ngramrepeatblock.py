# ops/microsoft_ngramrepeatblock.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def ngramrepeatblock_model_builder(op_type, cfg=None):
    input_ids = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [2, 5])
    scores = onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [2, 10])
    scores_out = onnx.helper.make_tensor_value_info("scores_out", onnx.TensorProto.FLOAT, [2, 10])

    node = onnx.helper.make_node(
        "NGramRepeatBlock",
        inputs=["input_ids", "scores"],
        outputs=["scores_out"],
        domain="com.microsoft",
        ngram_size=2
    )

    graph = onnx.helper.make_graph(
        [node],
        "NGramRepeatBlockGraph",
        [input_ids, scores],
        [scores_out]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def ngramrepeatblock_input_generator(session):
    input_ids = np.array([
        [1, 2, 3, 2, 3],
        [4, 5, 6, 5, 6]
    ], dtype=np.int64)
    scores = np.random.randn(2, 10).astype(np.float32)
    return {"input_ids": input_ids, "scores": scores}

SpecialModelBuilders["com.microsoft.NGramRepeatBlock"] = ngramrepeatblock_model_builder
SpecialInputGenerators["com.microsoft.NGramRepeatBlock"] = ngramrepeatblock_input_generator
