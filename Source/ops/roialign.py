# ops/roialign.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def roialign_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    rois = onnx.helper.make_tensor_value_info("rois", TensorProto.FLOAT, [1, 4])
    batch_indices = onnx.helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [1])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "RoiAlign",
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
        mode="avg",
        output_height=2,
        output_width=2,
        sampling_ratio=2,
        spatial_scale=1.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "roialign_graph",
        inputs=[x, rois, batch_indices],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def roialign_input_generator(session):
    X = np.arange(1, 65, dtype=np.float32).reshape(1, 1, 8, 8)
    rois = np.array([[1.0, 1.0, 6.0, 6.0]], dtype=np.float32)
    batch_indices = np.array([0], dtype=np.int64)
    return {"X": X, "rois": rois, "batch_indices": batch_indices}

SpecialModelBuilders["RoiAlign"] = roialign_model_builder
SpecialInputGenerators["RoiAlign"] = roialign_input_generator
