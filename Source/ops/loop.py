# ops/loop.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def make_loop_body():
    iter_count = onnx.helper.make_tensor_value_info("i", TensorProto.INT64, [])
    cond_in    = onnx.helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    acc_in     = onnx.helper.make_tensor_value_info("acc_in", TensorProto.FLOAT, [])

    cond_out = onnx.helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    acc_out  = onnx.helper.make_tensor_value_info("acc_out", TensorProto.FLOAT, [])

    one_const = onnx.helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])
    const_node = onnx.helper.make_node("Constant", [], ["one"], value=one_const)
    add_node = onnx.helper.make_node("Add", ["acc_in", "one"], ["acc_out"])
    identity_cond = onnx.helper.make_node("Identity", ["cond_in"], ["cond_out"])

    body = onnx.helper.make_graph(
        nodes=[const_node, add_node, identity_cond],
        name="loop_body",
        inputs=[iter_count, cond_in, acc_in],
        outputs=[cond_out, acc_out]
    )
    return body


def loop_model_builder(op_type, cfg=None):
    trip_count = onnx.helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    cond       = onnx.helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    init       = onnx.helper.make_tensor_value_info("acc_init", TensorProto.FLOAT, [])
    y          = onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Loop",
        inputs=["trip_count", "cond", "acc_init"],
        outputs=["y"],
        body=make_loop_body()
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="loop_graph",
        inputs=[trip_count, cond, init],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def loop_input_generator(session):
    return {
        "trip_count": np.array(5, dtype=np.int64),
        "cond": np.array(True, dtype=bool),
        "acc_init": np.array(0.0, dtype=np.float32)
    }

SpecialModelBuilders["Loop"] = loop_model_builder
SpecialInputGenerators["Loop"] = loop_input_generator
