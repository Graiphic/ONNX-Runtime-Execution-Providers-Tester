# ops/dequantizelinear.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def dequantizelinear_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
    x_scale = onnx.helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, [])
    x_zp = onnx.helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, [])
    y = onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DequantizeLinear",
        inputs=["x", "x_scale", "x_zero_point"],
        outputs=["y"],
        axis=1,
        block_size=0  # non utilisé ici (per-tensor)
    )

    graph = onnx.helper.make_graph(
        [node],
        "dequantizelinear_graph",
        [x, x_scale, x_zp],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["DequantizeLinear"] = dequantizelinear_model_builder

def dequantizelinear_input_generator(session):
    x_info, scale_info, zp_info = session.get_inputs()
    x = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
    scale = np.array(0.1, dtype=np.float32)
    zp = np.array(128, dtype=np.uint8)
    return {
        x_info.name: x,
        scale_info.name: scale,
        zp_info.name: zp
    }

SpecialInputGenerators["DequantizeLinear"] = dequantizelinear_input_generator
