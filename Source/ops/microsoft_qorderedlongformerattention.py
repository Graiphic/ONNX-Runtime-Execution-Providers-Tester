# ops/microsoft_qorderedlongformerattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qorderedlongformerattention_model_builder(op_type, cfg=None):
    num_heads = 2
    order_input = 1
    order_weight = 0
    order_output = 1
    order_global_weight = 0
    window = 2

    B, S, H = 1, 8, 16  # H=16 pour alignement matmul int8
    head_size = H // num_heads
    proj_size = 3 * H  # pour Q, K, V concaténés

    def make_input(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    inputs_info = [
        make_input("input", onnx.TensorProto.INT8, [B, S, H]),
        make_input("mask", onnx.TensorProto.FLOAT16, [B, S]),
        make_input("global", onnx.TensorProto.INT32, [B, S]),
    ]

    inits = []

    def add_const(name, array, dtype):
        tensor = onnx.helper.make_tensor(
            name=name,
            data_type=dtype,
            dims=array.shape,
            vals=array.flatten().tolist()
        )
        inits.append(tensor)

    # Scales
    scale_names = [
        "scale_input", "scale_weight", "scale_qkv_gemm",
        "scale_global_weight", "scale_global_gemm", "scale_output", "scale_bias"
    ]
    for name in scale_names:
        add_const(name, np.array(0.02, dtype=np.float32), onnx.TensorProto.FLOAT)

    # Weights and biases
    add_const("weight", np.random.randint(-128, 127, size=(H, proj_size), dtype=np.int8), onnx.TensorProto.INT8)
    add_const("bias", np.random.randn(proj_size).astype(np.float32), onnx.TensorProto.FLOAT)
    add_const("global_weight", np.random.randint(-128, 127, size=(H, proj_size), dtype=np.int8), onnx.TensorProto.INT8)
    add_const("global_bias", np.random.randn(proj_size).astype(np.float32), onnx.TensorProto.FLOAT)

    input_names = [
        "input", "scale_input", "weight", "scale_weight", "bias", "scale_bias",
        "scale_qkv_gemm", "mask", "global_weight", "scale_global_weight",
        "global_bias", "scale_global_gemm", "global", "scale_output"
    ]

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT8, None)

    node = onnx.helper.make_node(
        "QOrderedLongformerAttention",
        inputs=input_names,
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        order_input=order_input,
        order_weight=order_weight,
        order_output=order_output,
        order_global_weight=order_global_weight,
        window=window
    )

    graph = onnx.helper.make_graph(
        [node],
        "QOrderedLongformerAttentionGraph",
        inputs_info,
        [output],
        initializer=inits
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def qorderedlongformerattention_input_generator(session):
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.name == "mask":
            feed[inp.name] = np.random.rand(*shape).astype(np.float16)
        elif inp.name == "global":
            feed[inp.name] = np.random.randint(0, 2, size=shape, dtype=np.int32)
        else:
            feed[inp.name] = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    return feed

SpecialModelBuilders["com.microsoft.QOrderedLongformerAttention"] = qorderedlongformerattention_model_builder
SpecialInputGenerators["com.microsoft.QOrderedLongformerAttention"] = qorderedlongformerattention_input_generator
