# ops/microsoft_removepadding.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def removepadding_model_builder(op_type, cfg=None):
    B, S, H = 2, 5, 4  # batch, seq, hidden
    dtype = onnx.TensorProto.FLOAT

    x = onnx.helper.make_tensor_value_info("input", dtype, [B, S, H])
    token_count = onnx.helper.make_tensor_value_info("sequence_token_count", onnx.TensorProto.INT32, [B])
    y = onnx.helper.make_tensor_value_info("output", dtype, None)
    offset = onnx.helper.make_tensor_value_info("token_offset", onnx.TensorProto.INT32, [B, S])
    cum_seq = onnx.helper.make_tensor_value_info("cumulated_seq_len", onnx.TensorProto.INT32, [B + 1])
    max_len = onnx.helper.make_tensor_value_info("max_seq_len", onnx.TensorProto.INT32, [1])

    node = onnx.helper.make_node(
        "RemovePadding",
        inputs=["input", "sequence_token_count"],
        outputs=["output", "token_offset", "cumulated_seq_len", "max_seq_len"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "RemovePaddingGraph",
        [x, token_count],
        [y, offset, cum_seq, max_len]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def removepadding_input_generator(session):
    input_shape = [2, 5, 4]
    count_shape = [2]
    return {
        "input": np.random.randn(*input_shape).astype(np.float32),
        "sequence_token_count": np.array([3, 2], dtype=np.int32)
    }

SpecialModelBuilders["com.microsoft.RemovePadding"] = removepadding_model_builder
SpecialInputGenerators["com.microsoft.RemovePadding"] = removepadding_input_generator
