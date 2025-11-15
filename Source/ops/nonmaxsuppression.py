# ops/nonmaxsuppression.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def nms_model_builder(op_type, cfg=None):
    boxes = onnx.helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 6, 4])
    scores = onnx.helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 6])
    output = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, None)

    max_out = onnx.helper.make_tensor("max_output", TensorProto.INT64, [1], [3])
    iou = onnx.helper.make_tensor("iou_thresh", TensorProto.FLOAT, [1], [0.5])
    score = onnx.helper.make_tensor("score_thresh", TensorProto.FLOAT, [1], [0.0])

    node = onnx.helper.make_node(
        "NonMaxSuppression",
        inputs=["boxes", "scores", "max_output", "iou_thresh", "score_thresh"],
        outputs=["selected_indices"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "nms_graph",
        inputs=[boxes, scores],
        outputs=[output],
        initializer=[max_out, iou, score]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def nms_input_generator(session):
    boxes = np.array([[[0,0,1,1],[0,0,1.1,1.1],[0,0,0.9,0.9],
                       [1,1,2,2],[1,1,2.1,2.1],[1,1,1.9,1.9]]], dtype=np.float32)
    scores = np.array([[[0.9, 0.85, 0.8, 0.95, 0.9, 0.88]]], dtype=np.float32)
    return {"boxes": boxes, "scores": scores}

SpecialModelBuilders["NonMaxSuppression"] = nms_model_builder
SpecialInputGenerators["NonMaxSuppression"] = nms_input_generator
