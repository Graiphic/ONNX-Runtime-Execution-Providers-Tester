# ops/greedysearch.py

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def greedysearch_model_builder(op_type, cfg=None):
    """
    Modèle ONNX minimal pour l'opérateur com.microsoft::GreedySearch, 
    compatible avec CUDAExecutionProvider (qui impose un logits en 3D).
    """

    # 1) Entrées principales du graphe GreedySearch
    #    - input_ids : [batch_size=1, seq_len=5]
    #    - max_length: [1]
    #    - sorties : sequences [1, None]
    input_ids  = onnx.helper.make_tensor_value_info(
        "input_ids",  TensorProto.INT32, [1, 5]
    )
    max_length = onnx.helper.make_tensor_value_info(
        "max_length", TensorProto.INT32, [1]
    )
    sequences  = onnx.helper.make_tensor_value_info(
        "sequences", TensorProto.INT32, [1, None]
    )

    # 2) Sous-graphe "decoder" attendu par GreedySearch
    #    a) Définition des entrées du décodeur
    #       - input_ids    : [1, 1]
    #       - position_ids : [1, 1]
    #       - attention_mask: [1, 1]
    #       - past_0       : [2, 1, 1, 1, 1]  (2 = {key, value}, batch_size=1, num_heads=1, past_seq_len=1, head_size=1)
    decoder_inputs = [
        onnx.helper.make_tensor_value_info(
            "input_ids",      TensorProto.INT32, [1, 1]
        ),
        onnx.helper.make_tensor_value_info(
            "position_ids",   TensorProto.INT32, [1, 1]
        ),
        onnx.helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT32, [1, 1]
        ),
        onnx.helper.make_tensor_value_info(
            "past_0",         TensorProto.FLOAT, [2, 1, 1, 1, 1]
        ),
    ]

    #    b) Définition des sorties du décodeur
    #       - logits      : [1, 1, 10]  (batch_size=1, seq_len=1, vocab_size=10)
    #       - present_0   : [2, 1, 1, 1, 1] (nouveau past_state 5D)
    decoder_outputs = [
        onnx.helper.make_tensor_value_info(
            "logits",     TensorProto.FLOAT, [1, 1, 10]
        ),
        onnx.helper.make_tensor_value_info(
            "present_0",  TensorProto.FLOAT, [2, 1, 1, 1, 1]
        ),
    ]

    # 3) Création des nœuds internes du sous-graphe décodeur
    #    a) Constante pour produire logits "zéro" 3D : [1, 1, 10]
    const_logits = onnx.helper.make_tensor(
        name="const_logits",
        data_type=TensorProto.FLOAT,
        dims=[1, 1, 10],        # [batch_size=1, seq_len=1, vocab_size=10]
        vals=[0.0] * 10         # on remplit uniquement la dernière dimension (vocab_size)
        # ONNX répandtra ces 10 valeurs sur les dimensions [1,1,10]
    )
    node_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["logits"],
        value=const_logits
    )

    #    b) Identity pour propager past_0 vers present_0
    node_identity = onnx.helper.make_node(
        "Identity",
        inputs=["past_0"],
        outputs=["present_0"]
    )

    decoder_graph = onnx.helper.make_graph(
        [node_const, node_identity],
        "decoder_graph",
        decoder_inputs,
        decoder_outputs
    )

    # 4) Création du nœud GreedySearch principal
    node_greedy = onnx.helper.make_node(
        "GreedySearch",
        inputs=["input_ids", "max_length"],
        outputs=["sequences"],
        domain="com.microsoft",
        decoder=decoder_graph,
        decoder_start_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        model_type=0
        # Pas d’encoder ni init_decoder pour le minimal
    )

    # 5) Assemblage du graphe principal
    graph = onnx.helper.make_graph(
        [node_greedy],
        "GreedySearchGraph",
        [input_ids, max_length],
        [sequences]
    )

    # 6) Création du modèle ONNX avec imports d’opsets
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION),     # ONNX standard
            onnx.helper.make_operatorsetid("com.microsoft", 1)           # com.microsoft v1
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    # 7) Vérification du modèle (check_model soulèvera une exception en cas d’incohérence)
    onnx.checker.check_model(model)
    return model

def greedysearch_input_generator(session):
    """
    Génère les entrées pour GreedySearch :
    - input_ids  : np.array([[10, 20, 30, 40, 50]], dtype=int32), shape [1,5]
    - max_length : np.array([10], dtype=int32), shape [1]
    Les autres tenseurs (attention_mask, position_ids, past_0) sont créés par ONNXRuntime ou considérés par défaut.
    """
    feed = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = np.array([[10, 20, 30, 40, 50]], dtype=np.int32)
        elif inp.name == "max_length":
            feed[inp.name] = np.array([10], dtype=np.int32)
    return feed

# Enregistrement pour OpTest
SpecialModelBuilders["com.microsoft.GreedySearch"] = greedysearch_model_builder
SpecialInputGenerators["com.microsoft.GreedySearch"] = greedysearch_input_generator
