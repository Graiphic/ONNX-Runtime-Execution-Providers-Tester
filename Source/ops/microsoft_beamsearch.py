# ops/beamsearch.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def beamsearch_model_builder(op_type, cfg=None):
    batch_size = 1
    sequence_length = 5
    vocab_size = 50257
    max_length = 10
    num_beams = 4
    num_return_sequences = 2

    # 1) Déclaration des inputs du graphe principal
    input_ids_vi = onnx.helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, [batch_size, sequence_length]
    )
    max_length_vi = onnx.helper.make_tensor_value_info(
        "max_length", TensorProto.INT32, [1]
    )
    num_beams_vi = onnx.helper.make_tensor_value_info(
        "num_beams", TensorProto.INT32, [1]
    )
    num_return_sequences_vi = onnx.helper.make_tensor_value_info(
        "num_return_sequences", TensorProto.INT32, [1]
    )

    # 2) Initializers obligatoires pour BeamSearch
    #    • min_length   : INT32 [1] = 0
    const_min_length = onnx.helper.make_tensor(
        name="min_length",
        data_type=TensorProto.INT32,
        dims=(1,),
        vals=[0]
    )
    #    • length_penalty   : FLOAT [1] = 1.0
    const_length_penalty = onnx.helper.make_tensor(
        name="length_penalty",
        data_type=TensorProto.FLOAT,
        dims=(1,),
        vals=[1.0]
    )
    #    • repetition_penalty : FLOAT [1] = 1.0
    const_repetition_penalty = onnx.helper.make_tensor(
        name="repetition_penalty",
        data_type=TensorProto.FLOAT,
        dims=(1,),
        vals=[1.0]
    )

    # 3) Déclaration de la sortie du graphe principal
    sequences_vi = onnx.helper.make_tensor_value_info(
        "sequences", TensorProto.INT32, [batch_size, num_return_sequences, max_length]
    )

    # 4) Construction du sous‐graphe DecoderGraph pour CPU
    #
    #    Rappel : BeamSearch pour CPU attend un petit sous-graphe « Decoder » qui
    #    doit renvoyer :
    #      - logits : FLOAT 3D de forme [batch_size, 1, vocab_size]
    #      - present_0 : FLOAT 5D de forme [2, batch_size, num_heads, seq_len, head_size]
    #
    #    Ici on se contente de deux nœuds :
    #      • un Constant pour produire un tensor logits rempli de zéros
    #      • un Identity pour propager « past_0 → present_0 » (forme 5D)
    #
    #    a) value_info des inputs du sous-graphe
    decoder_input_ids      = onnx.helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, [batch_size, 1]
    )
    decoder_position_ids   = onnx.helper.make_tensor_value_info(
        "position_ids", TensorProto.INT32, [batch_size, 1]
    )
    decoder_attention_mask = onnx.helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, [batch_size, 1]
    )
    #    • past_0 : FLOAT 5D (on choisit num_heads=1, seq_len=1, head_size=1)
    decoder_past_0 = onnx.helper.make_tensor_value_info(
        "past_0", TensorProto.FLOAT, [2, batch_size, 1, 1, 1]
    )

    #    b) value_info des outputs du sous-graphe
    decoder_logits    = onnx.helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [batch_size, 1, vocab_size]
    )
    decoder_present_0 = onnx.helper.make_tensor_value_info(
        "present_0", TensorProto.FLOAT, [2, batch_size, 1, 1, 1]
    )

    #    c) initializer pour logits 3D rempli de zéros
    zero_logits = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)
    const_logits_init = onnx.helper.make_tensor(
        name="const_logits_tensor",
        data_type=TensorProto.FLOAT,
        dims=(batch_size, 1, vocab_size),
        vals=zero_logits.flatten().tolist()
    )
    #    → ce tensor servira directement de sortie « logits » dans le sous-graphe

    #    d) nœud Constant (cpu) : génère « logits » (= const_logits_init)
    decoder_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["logits"],
        value=const_logits_init
    )

    #    e) nœud Identity : « past_0 → present_0 »
    decoder_identity_node = onnx.helper.make_node(
        "Identity",
        inputs=["past_0"],
        outputs=["present_0"]
    )

    #    f) création du graphe dans le domaine ONNX standard
    decoder_graph = onnx.helper.make_graph(
        [decoder_const_node, decoder_identity_node],       # nodes
        "DecoderGraph",                                    # nom du sous-graphe
        [decoder_input_ids, decoder_position_ids,           # inputs 
         decoder_attention_mask, decoder_past_0],
        [decoder_logits, decoder_present_0],                # outputs
        initializer=[const_logits_init]                     # initializer pour logits
    )

    # 5) Noeud BeamSearch du graphe principal
    beamsearch_node = onnx.helper.make_node(
        "BeamSearch",
        inputs=[
            "input_ids",          # 0. input_ids
            "max_length",         # 1. max_length
            "min_length",         # 2. min_length (initializer)
            "num_beams",          # 3. num_beams
            "num_return_sequences",  # 4. num_return_sequences
            "length_penalty",     # 5. length_penalty (initializer)
            "repetition_penalty"  # 6. repetition_penalty (initializer)
        ],
        outputs=["sequences"],
        domain="com.microsoft",
        decoder=decoder_graph,
        eos_token_id=50256,
        pad_token_id=0
    )

    # 6) Assemblage du graphe principal
    graph = onnx.helper.make_graph(
        nodes=[beamsearch_node],
        name="BeamSearchGraph",
        inputs=[input_ids_vi, max_length_vi, num_beams_vi, num_return_sequences_vi],
        outputs=[sequences_vi],
        initializer=[const_min_length, const_length_penalty, const_repetition_penalty]
    )

    # 7) Création du modèle avec les bons opset_imports
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            # ONNX standard pour les ops de base (Constant, Identity, etc.)
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION),
            # Domaine com.microsoft pour BeamSearch (version 1)
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def beamsearch_input_generator(session):
    batch_size = 1
    sequence_length = 5
    vocab_size = 50257
    max_length = 10
    num_beams = 4
    num_return_sequences = 2

    input_ids = np.random.randint(
        0, vocab_size, size=(batch_size, sequence_length), dtype=np.int32
    )
    max_length_tensor = np.array([max_length], dtype=np.int32)
    num_beams_tensor = np.array([num_beams], dtype=np.int32)
    num_return_sequences_tensor = np.array([num_return_sequences], dtype=np.int32)

    return {
        "input_ids": input_ids,
        "max_length": max_length_tensor,
        "num_beams": num_beams_tensor,
        "num_return_sequences": num_return_sequences_tensor
        # min_length, length_penalty, repetition_penalty sont fournis par initializers
    }

SpecialModelBuilders["com.microsoft.BeamSearch"] = beamsearch_model_builder
SpecialInputGenerators["com.microsoft.BeamSearch"] = beamsearch_input_generator
