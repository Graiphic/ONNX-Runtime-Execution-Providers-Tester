# ops/reversesequence.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def reversesequence_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 3])
    seq_lens = onnx.helper.make_tensor_value_info("sequence_lens", TensorProto.INT64, [3])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "ReverseSequence",
        inputs=["X", "sequence_lens"],
        outputs=["Y"],
        time_axis=0,
        batch_axis=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "reversesequence_graph",
        inputs=[x, seq_lens],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def reversesequence_input_generator(session):
    x = np.arange(4 * 3).reshape(4, 3).astype(np.float32)  # shape [time=4, batch=3]
    seq_lens = np.array([4, 3, 2], dtype=np.int64)
    return {"X": x, "sequence_lens": seq_lens}

SpecialModelBuilders["ReverseSequence"] = reversesequence_model_builder
SpecialInputGenerators["ReverseSequence"] = reversesequence_input_generator
