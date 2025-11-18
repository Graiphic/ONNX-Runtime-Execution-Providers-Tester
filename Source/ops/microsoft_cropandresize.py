# ops/cropandresize.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def cropandresize_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None, None, None])  # (N, C, H, W)
    rois = onnx.helper.make_tensor_value_info("rois", onnx.TensorProto.FLOAT, [None, 4])
    batch_indices = onnx.helper.make_tensor_value_info("batch_indices", onnx.TensorProto.INT32, [None])
    crop_size = onnx.helper.make_tensor_value_info("crop_size", onnx.TensorProto.INT32, [2])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "CropAndResize",
        inputs=["X", "rois", "batch_indices", "crop_size"],
        outputs=["Y"],
        domain="com.microsoft",
        mode="bilinear",
        extrapolation_value=0.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "CropAndResizeGraph",
        [X, rois, batch_indices, crop_size],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def cropandresize_input_generator(session):
    N, C, H, W = 2, 3, 32, 32
    num_rois = 4
    crop_height, crop_width = 10, 10

    X = np.random.randn(N, C, H, W).astype(np.float32)
    rois = np.random.rand(num_rois, 4).astype(np.float32)  # Normalized [y1, x1, y2, x2]
    rois[:, 2:] = np.maximum(rois[:, :2] + 0.1, rois[:, 2:])  # Ensure y2 > y1, x2 > x1
    batch_indices = np.random.randint(0, N, size=(num_rois,), dtype=np.int32)
    crop_size = np.array([crop_height, crop_width], dtype=np.int32)

    return {
        "X": X,
        "rois": rois,
        "batch_indices": batch_indices,
        "crop_size": crop_size
    }

SpecialModelBuilders["com.microsoft.CropAndResize"] = cropandresize_model_builder
SpecialInputGenerators["com.microsoft.CropAndResize"] = cropandresize_input_generator
