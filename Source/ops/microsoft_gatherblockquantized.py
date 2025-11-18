# ops/gatherblockquantized.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def gatherblockquantized_model_builder(op_type, cfg=None):
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.UINT8, [64, 8])
    indices_info = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [3])
    scales_info = onnx.helper.make_tensor_value_info("scales", onnx.TensorProto.FLOAT, [64, 8])
    #zero_points_info = onnx.helper.make_tensor_value_info("zero_points", onnx.TensorProto.UINT8, [64, 8])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "GatherBlockQuantized",
        inputs=["data", "indices", "scales"],
        outputs=["output"],
        domain="com.microsoft",
        gather_axis=0,
        quantize_axis=0,
        block_size=16
    )

    graph = onnx.helper.make_graph(
        [node],
        "GatherBlockQuantizedGraph",
        [data_info, indices_info, scales_info],
        [output_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def gatherblockquantized_input_generator(session):
    data = np.random.randint(0, 256, size=(64, 8), dtype=np.uint8)
    indices = np.random.randint(0, 64, size=(3,), dtype=np.int64)
    scales = np.vstack([np.full((16, 8), s, dtype=np.float32) for s in np.random.rand(4)])
    #zero_points = np.vstack([np.full((16, 8), 128, dtype=np.uint8) for _ in range(4)])

    return {
        "data": data,
        "indices": indices,
        "scales": scales,
        #"zero_points": zero_points
    }

SpecialModelBuilders["com.microsoft.GatherBlockQuantized"] = gatherblockquantized_model_builder
SpecialInputGenerators["com.microsoft.GatherBlockQuantized"] = gatherblockquantized_input_generator
