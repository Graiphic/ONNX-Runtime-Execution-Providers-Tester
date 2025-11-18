# ops/col2im.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def col2im_model_builder(op_type, cfg=None):
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 9, 25])
    image_shape = onnx.helper.make_tensor_value_info("image_shape", onnx.TensorProto.INT64, [2])
    block_shape = onnx.helper.make_tensor_value_info("block_shape", onnx.TensorProto.INT64, [2])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Col2Im",
        inputs=["input", "image_shape", "block_shape"],
        outputs=["output"],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1]
    )

    graph = onnx.helper.make_graph(
        [node],
        "col2im_graph",
        [input_tensor, image_shape, block_shape],
        [output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Col2Im"] = col2im_model_builder

def col2im_input_generator(session):
    input_info, image_shape_info, block_shape_info = session.get_inputs()
    input_data = np.random.rand(1, 9, 25).astype(np.float32)  # 3x3 blocks, 5x5 grid
    image_shape = np.array([5, 5], dtype=np.int64)
    block_shape = np.array([3, 3], dtype=np.int64)
    return {
        input_info.name: input_data,
        image_shape_info.name: image_shape,
        block_shape_info.name: block_shape
    }

SpecialInputGenerators["Col2Im"] = col2im_input_generator
