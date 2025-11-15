# ops/rnn.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def rnn_model_builder(op_type, cfg=None):
    seq_len, batch_size, input_size, hidden_size = 4, 2, 3, 5
    num_directions = 1  # 'forward'

    X = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_len, batch_size, input_size])
    W = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, [num_directions, hidden_size, input_size])
    R = onnx.helper.make_tensor_value_info("R", TensorProto.FLOAT, [num_directions, hidden_size, hidden_size])
    B = onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, [num_directions, 2 * hidden_size])
    initial_h = onnx.helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size])

    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    Y_h = onnx.helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RNN",
        inputs=["X", "W", "R", "B", "", "initial_h"],
        outputs=["Y", "Y_h"],
        hidden_size=hidden_size,
        direction="forward",
        activations=["Tanh"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "rnn_graph",
        inputs=[X, W, R, B, initial_h],
        outputs=[Y, Y_h]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def rnn_input_generator(session):
    seq_len, batch_size, input_size, hidden_size = 4, 2, 3, 5
    num_directions = 1

    X = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
    W = np.random.randn(num_directions, hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 2 * hidden_size).astype(np.float32)
    initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)

    return {"X": X, "W": W, "R": R, "B": B, "initial_h": initial_h}

SpecialModelBuilders["RNN"] = rnn_model_builder
SpecialInputGenerators["RNN"] = rnn_input_generator
