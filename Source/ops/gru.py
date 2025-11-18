# ops/gru.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gru_model_builder(op_type, cfg=None):
    # Dimensions : seq_length=4, batch_size=2, input_size=3, hidden_size=5
    seq_length = 4
    batch_size = 2
    input_size = 3
    hidden_size = 5
    num_directions = 1  # Unidirectionnel

    # Entrées
    X = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_length, batch_size, input_size])
    W = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, [num_directions, 3 * hidden_size, input_size])
    R = onnx.helper.make_tensor_value_info("R", TensorProto.FLOAT, [num_directions, 3 * hidden_size, hidden_size])
    B = onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, [num_directions, 6 * hidden_size])
    initial_h = onnx.helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size])

    # Sorties
    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq_length, num_directions, batch_size, hidden_size])
    Y_h = onnx.helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [num_directions, batch_size, hidden_size])

    # Nœud GRU
    node = onnx.helper.make_node(
        "GRU",
        inputs=["X", "W", "R", "B", "", "initial_h"],  # L'entrée sequence_lens est omise
        outputs=["Y", "Y_h"],
        #outputs=["Y"],
        hidden_size=hidden_size,
        direction="forward"
    )

    # Graph
    graph = onnx.helper.make_graph(
        [node],
        "gru_graph",
        [X, W, R, B, initial_h],
        [Y, Y_h]
        #[Y]
    )

    # Modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["GRU"] = gru_model_builder

def gru_input_generator(session):
    # Récupération des informations sur les entrées
    input_infos = {inp.name: inp for inp in session.get_inputs()}
    seq_length, batch_size, input_size = input_infos["X"].shape
    hidden_size = input_infos["initial_h"].shape[2]
    num_directions = 1  # Unidirectionnel

    # Génération des données d'entrée
    X = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    W = np.random.randn(num_directions, 3 * hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, 3 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 6 * hidden_size).astype(np.float32)
    initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)

    return {
        "X": X,
        "W": W,
        "R": R,
        "B": B,
        "initial_h": initial_h
    }

SpecialInputGenerators["GRU"] = gru_input_generator
