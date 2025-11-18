# ops/attnlstm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def attnlstm_model_builder(op_type, cfg=None):
    seq_len, batch_size, input_size = 5, 2, 4
    hidden_size = 3
    num_directions = 1

    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [seq_len, batch_size, input_size])
    W = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [num_directions, 4 * hidden_size, input_size])
    R = onnx.helper.make_tensor_value_info("R", onnx.TensorProto.FLOAT, [num_directions, 4 * hidden_size, hidden_size])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [seq_len, num_directions, batch_size, hidden_size])

    node = onnx.helper.make_node(
        "AttnLSTM",
        inputs=["X", "W", "R"],
        outputs=["Y"],
        domain="com.microsoft",
        hidden_size=hidden_size,
        direction="forward"
    )

    graph = onnx.helper.make_graph(
        [node],
        "MinimalAttnLSTM",
        [X, W, R],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def attnlstm_input_generator(session):
    seq_len, batch_size, input_size, hidden_size = 5, 2, 4, 3
    return {
        "X": np.random.rand(seq_len, batch_size, input_size).astype(np.float32),
        "W": np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32),
        "R": np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32),
    }

SpecialModelBuilders["com.microsoft.AttnLSTM"] = attnlstm_model_builder
SpecialInputGenerators["com.microsoft.AttnLSTM"] = attnlstm_input_generator
