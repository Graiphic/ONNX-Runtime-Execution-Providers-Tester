# ops/microsoft_whisperbeamsearch.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto, OperatorSetIdProto, numpy_helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

# ------------------------------------------------------------------
LAYERS        = 24                      # 2 × 24 = 48 sorties pour l'encoder
VOCAB_SIZE    = 20
# Initialisers communs
ZERO_F32      = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="zero")
SHAPE4D_I64   = numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="shape4D")
# ------------------------------------------------------------------
# 1. ENCODER SUBGRAPH  (2 inputs, 48 outputs)
# ------------------------------------------------------------------
e_in_ids  = helper.make_tensor_value_info("encoder_input_ids",      TensorProto.FLOAT, [1, None, None])
e_in_mask = helper.make_tensor_value_info("encoder_attention_mask", TensorProto.INT32,  [1, None])

enc_outputs = []
enc_const_nodes = []
for l in range(LAYERS):
    key_name  = f"present_key_cross_{l}"
    val_name  = f"present_value_cross_{l}"
    enc_outputs.extend([
        helper.make_tensor_value_info(key_name, TensorProto.FLOAT, [1, None, None, None]),
        helper.make_tensor_value_info(val_name, TensorProto.FLOAT, [1, None, None, None])
    ])
    # Génère un 4-D constant de zéros
    enc_const_nodes.extend([
        helper.make_node("ConstantOfShape", ["shape4D"], [key_name], value=ZERO_F32),
        helper.make_node("ConstantOfShape", ["shape4D"], [val_name], value=ZERO_F32)
    ])

encoder_graph = helper.make_graph(
    enc_const_nodes,
    "EncoderGraph",
    [e_in_ids, e_in_mask],
    enc_outputs,
    initializer=[SHAPE4D_I64]
)

# ------------------------------------------------------------------
# 2. DECODER SUBGRAPH (inchangé : 98 sorties)
# ------------------------------------------------------------------
d_in_ids    = helper.make_tensor_value_info("decoder_input_ids",    TensorProto.INT32,  [1])
d_in_hidden = helper.make_tensor_value_info("encoder_hidden_states",TensorProto.FLOAT, [1, None, None])

logits_out  = helper.make_tensor_value_info("logits",               TensorProto.FLOAT, [1, 1, VOCAB_SIZE])
dec_hidden  = helper.make_tensor_value_info("decoder_hidden_states",TensorProto.FLOAT, [1, None, None])

dec_outputs = [logits_out, dec_hidden]
dec_nodes   = [helper.make_node("ConstantOfShape", ["shape4D"], ["dummy"], value=ZERO_F32)]  # placeholder

# --- extrait uniquement de la section "DECODER SUBGRAPH" ---

dec_logits_init = numpy_helper.from_array(
    np.zeros((1, 1, VOCAB_SIZE), dtype=np.float32), name="dec_logits_init"
)

# nœud qui génère vraiment la sortie 'logits'
id_logits   = helper.make_node("Identity", ["dec_logits_init"], ["logits"])
id_hidden   = helper.make_node("Identity", ["encoder_hidden_states"], ["decoder_hidden_states"])
shape4d     = helper.make_node("Identity", ["shape4D"], ["shape4D_pass"])  # juste pour consommer l’init

dec_nodes   = [id_logits, id_hidden]

# ConstantOfShape → Unsqueeze 4-D pour chaque PKV
for l in range(LAYERS):
    for suf in ("key_self", "value_self", "key_cross", "value_cross"):
        name = f"present_{suf}_{l}"
        raw  = f"raw_{name}"
        dec_nodes.extend([
            helper.make_node("ConstantOfShape", ["shape4D_pass"], [raw], value=ZERO_F32),
            helper.make_node("Identity", [raw], [name])  # 4-D direct
        ])

decoder_graph = helper.make_graph(
    dec_nodes,
    "DecoderGraph",
    [d_in_ids, d_in_hidden],
    [logits_out, dec_hidden, *dec_outputs[2:]],
    initializer=[SHAPE4D_I64, dec_logits_init]
)


# ------------------------------------------------------------------
# 3. OPSET IMPORTS
# ------------------------------------------------------------------
opset_imports = [
    helper.make_operatorsetid("", 17),
    helper.make_operatorsetid("com.microsoft", 1)
]

# ------------------------------------------------------------------
# 4. GRAPH PRINCIPAL (10 inputs comme précédemment)
# ------------------------------------------------------------------
main_inputs = [
    helper.make_tensor_value_info("encoder_input_ids",      TensorProto.FLOAT, [1, None, None]),
    helper.make_tensor_value_info("max_length",             TensorProto.INT32, [1]),
    helper.make_tensor_value_info("min_length",             TensorProto.INT32, [1]),
    helper.make_tensor_value_info("num_beams",              TensorProto.INT32, [1]),
    helper.make_tensor_value_info("num_return_sequences",   TensorProto.INT32, [1]),
    helper.make_tensor_value_info("length_penalty",         TensorProto.FLOAT, [1]),
    helper.make_tensor_value_info("repetition_penalty",     TensorProto.FLOAT, [1]),
    helper.make_tensor_value_info("vocab_mask",             TensorProto.INT32, [VOCAB_SIZE]),
    helper.make_tensor_value_info("prefix_vocab_mask",      TensorProto.INT32, [1, VOCAB_SIZE]),
    helper.make_tensor_value_info("attention_mask",         TensorProto.INT32, [1, None]),
]

seqs_out = helper.make_tensor_value_info("sequences", TensorProto.INT32, [None, None, None])

wb_node = helper.make_node(
    "WhisperBeamSearch",
    inputs=[i.name for i in main_inputs],
    outputs=["sequences"],
    domain="com.microsoft",
    encoder=encoder_graph,
    decoder=decoder_graph,
    eos_token_id=50256,
    pad_token_id=50257,
    decoder_start_token_id=50258,
    beginning_timestamp_token_id=50360,
    no_timestamps_token_id=50361,
    model_type=2
)

main_graph = helper.make_graph(
    [wb_node],
    "WhisperBeamSearchGraph",
    inputs=main_inputs,
    outputs=[seqs_out]
)

model_proto = helper.make_model(
    main_graph,
    opset_imports=opset_imports,
    ir_version=ONNX_RUNTIME_IR_VERSION
)

# ------------------------------------------------------------------
# 5. REGISTRATION POUR LE HARNESS
# ------------------------------------------------------------------
def whisper_model_builder(op_type, cfg=None):
    return model_proto


def whisper_input_gen(session):
    seq_len, feat = 1500, 80
    return {
        "encoder_input_ids":      np.random.rand(1, feat, seq_len).astype(np.float32),
        "max_length":             np.array([16],  dtype=np.int32),
        "min_length":             np.array([1],   dtype=np.int32),
        "num_beams":              np.array([4],   dtype=np.int32),
        "num_return_sequences":   np.array([1],   dtype=np.int32),
        "length_penalty":         np.array([1.0], dtype=np.float32),
        "repetition_penalty":     np.array([1.0], dtype=np.float32),
        "vocab_mask":             np.ones((VOCAB_SIZE,),   dtype=np.int32),
        "prefix_vocab_mask":      np.ones((1, VOCAB_SIZE), dtype=np.int32),
        "attention_mask":         np.ones((1, seq_len),    dtype=np.int32)
    }

SpecialModelBuilders["com.microsoft.WhisperBeamSearch"]   = whisper_model_builder
SpecialInputGenerators["com.microsoft.WhisperBeamSearch"] = whisper_input_gen
