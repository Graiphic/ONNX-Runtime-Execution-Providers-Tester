# ops/nllloss.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def nllloss_model_builder(op_type, cfg=None):
    input = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5])
    target = onnx.helper.make_tensor_value_info("target", TensorProto.INT64, [3])
    loss = onnx.helper.make_tensor_value_info("loss", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=["X", "target"],
        outputs=["loss"],
        reduction="mean"
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="nllloss_graph",
        inputs=[input, target],
        outputs=[loss]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def nllloss_input_generator(session):
    input = np.random.rand(3, 5).astype(np.float32)
    input = np.log(input / np.sum(input, axis=1, keepdims=True))  # log-softmax-like
    target = np.random.randint(0, 5, size=(3,), dtype=np.int64)
    return {"X": input, "target": target}

SpecialModelBuilders["NegativeLogLikelihoodLoss"] = nllloss_model_builder
SpecialInputGenerators["NegativeLogLikelihoodLoss"] = nllloss_input_generator
