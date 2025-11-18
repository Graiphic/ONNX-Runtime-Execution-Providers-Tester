# ops/dequantizewithorder.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dequantizewithorder_model_builder(op_type, cfg=None):
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [2, 4])
    scale_info = onnx.helper.make_tensor_value_info("scale_input", onnx.TensorProto.FLOAT, [])

    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DequantizeWithOrder",
        inputs=["input", "scale_input"],
        outputs=["output"],
        domain="com.microsoft",
        order_input=1,   # par exemple : RowMajor
        order_output=1,  # par exemple : RowMajor
        to=1             # float32 (TensorProto_DataType_FLOAT)
    )

    graph = onnx.helper.make_graph(
        [node],
        "DequantizeWithOrderGraph",
        [input_info, scale_info],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def dequantizewithorder_input_generator(session):
    shape = (2, 4)
    input_tensor = np.random.randint(-128, 128, size=shape, dtype=np.int8)
    scale = np.array(0.1, dtype=np.float32)

    return {
        "input": input_tensor,
        "scale_input": scale
    }

SpecialModelBuilders["com.microsoft.DequantizeWithOrder"] = dequantizewithorder_model_builder
SpecialInputGenerators["com.microsoft.DequantizeWithOrder"] = dequantizewithorder_input_generator
