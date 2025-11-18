# ops/castlike.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def castlike_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 2])
    t = onnx.helper.make_tensor_value_info("T", onnx.TensorProto.INT32, [2, 2])
    y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.INT32, None)

    node = onnx.helper.make_node(
        "CastLike",
        inputs=["X", "T"],
        outputs=["Y"]
    )

    graph = onnx.helper.make_graph(
        [node],
        "castlike_graph",
        [x, t],
        [y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["CastLike"] = castlike_model_builder

def castlike_input_generator(session):
    x_info, t_info = session.get_inputs()
    shape = [d or 1 for d in x_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    t = np.random.randint(0, 10, size=shape).astype(np.int32)
    return {x_info.name: x, t_info.name: t}

SpecialInputGenerators["CastLike"] = castlike_input_generator
