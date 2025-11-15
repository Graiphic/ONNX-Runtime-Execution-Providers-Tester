# ops/microsoft_qattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qattention_model_builder(op_type, cfg=None):
    batch_size = 2
    sequence_length = 4
    input_hidden_size = 8
    num_heads = 2
    head_size = 4
    hidden_size = num_heads * head_size

    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT8, [batch_size, sequence_length, input_hidden_size])
    weight = onnx.helper.make_tensor_value_info("weight", onnx.TensorProto.UINT8, [input_hidden_size, 3 * hidden_size])
    bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3 * hidden_size])
    input_scale = onnx.helper.make_tensor_value_info("input_scale", onnx.TensorProto.FLOAT, [])
    weight_scale = onnx.helper.make_tensor_value_info("weight_scale", onnx.TensorProto.FLOAT, [])

    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "QAttention",
        inputs=["input", "weight", "bias", "input_scale", "weight_scale"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads
    )

    graph = onnx.helper.make_graph(
        [node],
        "QAttentionGraph",
        [input_tensor, weight, bias, input_scale, weight_scale],
        [output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def qattention_input_generator(session):
    batch_size = 2
    sequence_length = 4
    input_hidden_size = 8
    num_heads = 2
    head_size = 4
    hidden_size = num_heads * head_size

    inputs = {
        "input": np.random.randint(0, 255, size=(batch_size, sequence_length, input_hidden_size), dtype=np.uint8),
        "weight": np.random.randint(0, 255, size=(input_hidden_size, 3 * hidden_size), dtype=np.uint8),
        "bias": np.random.randn(3 * hidden_size).astype(np.float32),
        "input_scale": np.array(0.02, dtype=np.float32),
        "weight_scale": np.array(0.01, dtype=np.float32)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QAttention"] = qattention_model_builder
SpecialInputGenerators["com.microsoft.QAttention"] = qattention_input_generator
