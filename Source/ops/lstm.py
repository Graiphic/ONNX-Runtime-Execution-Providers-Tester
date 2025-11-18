# ops/lstm.py
import numpy as np
import onnx
import onnx.helper
from utils import (
    SpecialModelBuilders,
    SpecialInputGenerators,
    ONNX_RUNTIME_IR_VERSION,
    ONNX_OPSET_VERSION,
)

def lstm_model_builder(op_type, cfg=None):
    seq_length = 5
    batch_size = 2
    input_size = 3
    hidden_size = 4
    num_directions = 1  # 2 pour bidirectionnel

    # Définition des entrées
    X = onnx.helper.make_tensor_value_info(
        "X", onnx.TensorProto.FLOAT, [seq_length, batch_size, input_size]
    )
    W = onnx.helper.make_tensor_value_info(
        "W", onnx.TensorProto.FLOAT, [num_directions, 4 * hidden_size, input_size]
    )
    R = onnx.helper.make_tensor_value_info(
        "R", onnx.TensorProto.FLOAT, [num_directions, 4 * hidden_size, hidden_size]
    )
    B = onnx.helper.make_tensor_value_info(
        "B", onnx.TensorProto.FLOAT, [num_directions, 8 * hidden_size]
    )
    initial_h = onnx.helper.make_tensor_value_info(
        "initial_h", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
    )
    initial_c = onnx.helper.make_tensor_value_info(
        "initial_c", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
    )

    # Définition des sorties
    Y = onnx.helper.make_tensor_value_info(
        "Y", onnx.TensorProto.FLOAT, [seq_length, num_directions, batch_size, hidden_size]
    )
    Y_h = onnx.helper.make_tensor_value_info(
        "Y_h", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
    )
    Y_c = onnx.helper.make_tensor_value_info(
        "Y_c", onnx.TensorProto.FLOAT, [num_directions, batch_size, hidden_size]
    )

    # Création du nœud LSTM
    node = onnx.helper.make_node(
        "LSTM",
        inputs=["X", "W", "R", "B", "", "initial_h", "initial_c", ""],
        outputs=["Y", "Y_h", "Y_c"],
        #outputs=["Y", "Y_h"],
        hidden_size=hidden_size,
        direction="forward",
        input_forget = 0,
    )

    # Construction du graphe
    graph = onnx.helper.make_graph(
        nodes=[node],
        name="lstm_graph",
        inputs=[X, W, R, B, initial_h, initial_c],
        outputs=[Y, Y_h, Y_c],
        #outputs=[Y, Y_h],
    )

    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)
        ],
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def lstm_input_generator(session):
    inputs = {inp.name: inp for inp in session.get_inputs()}
    seq_length, batch_size, input_size = inputs["X"].shape
    hidden_size = inputs["R"].shape[2]
    num_directions = inputs["R"].shape[0]

    X = np.random.rand(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.rand(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.rand(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)
    initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)

    return {
        "X": X,
        "W": W,
        "R": R,
        "B": B,
        "initial_h": initial_h,
        "initial_c": initial_c,
    }

SpecialModelBuilders["LSTM"] = lstm_model_builder
SpecialInputGenerators["LSTM"] = lstm_input_generator
