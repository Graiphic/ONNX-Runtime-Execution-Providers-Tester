# ops/microsoft_tokenizer.py
import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def tokenizer_model_builder(op_type, cfg=None):
    X = helper.make_tensor_value_info("X", TensorProto.STRING, [2])  # ex. ["Hello World", "GPT rocks"]
    Y = helper.make_tensor_value_info("Y", TensorProto.STRING, None)  # shape [2, max_tokens]

    node = helper.make_node(
        "Tokenizer",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
        mark=1,
        mincharnum=1,
        pad_value="<pad>",
        separators=[" "]
    )

    graph = helper.make_graph(
        [node],
        "TokenizerGraph",
        [X],
        [Y]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def tokenizer_input_generator(session):
    inputs = session.get_inputs()
    input_name = inputs[0].name
    data = np.array(["Hello world", "GPT rocks!"], dtype=np.object_)
    return {input_name: data}

SpecialModelBuilders["com.microsoft.Tokenizer"] = tokenizer_model_builder
SpecialInputGenerators["com.microsoft.Tokenizer"] = tokenizer_input_generator
