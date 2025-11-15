# ops/microsoft_restorepadding.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def restorepadding_model_builder(op_type, cfg=None):
    B, S, H = 2, 5, 4      # batch, sequence, hidden_size
    total_tokens = 5       # tokens utiles
    dtype = onnx.TensorProto.FLOAT

    inp = onnx.helper.make_tensor_value_info("input", dtype, [total_tokens, H])
    offset = onnx.helper.make_tensor_value_info("token_offset", onnx.TensorProto.INT32, [B, S])
    out = onnx.helper.make_tensor_value_info("output", dtype, [B, S, H])

    node = onnx.helper.make_node(
        "RestorePadding",
        inputs=["input", "token_offset"],
        outputs=["output"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "RestorePaddingGraph",
        [inp, offset],
        [out]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def restorepadding_input_generator(session):
    total_tokens, H = 5, 4
    B, S = 2, 5
    return {
        "input": np.random.randn(total_tokens, H).astype(np.float32),
        "token_offset": np.array([[0, 1, 2, 5, 6], [3, 4, 5, 6, 7]], dtype=np.int32)
    }

SpecialModelBuilders["com.microsoft.RestorePadding"] = restorepadding_model_builder
SpecialInputGenerators["com.microsoft.RestorePadding"] = restorepadding_input_generator
