# ops/if.py
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def make_branch_graph(output_value, output_name):
    const_tensor = onnx.helper.make_tensor(
        name="const",
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=[output_value],
    )
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["out"],
        value=const_tensor,
    )
    output = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [])
    graph = onnx.helper.make_graph(
        nodes=[const_node],
        name="branch_graph",
        inputs=[],
        outputs=[output],
    )
    return graph

def if_model_builder(op_type, cfg=None):
    cond_input = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    then_branch = make_branch_graph(1.0, "Y")
    else_branch = make_branch_graph(0.0, "Y")

    node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["Y"],
        then_branch=then_branch,
        else_branch=else_branch,
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="if_graph",
        inputs=[cond_input],
        outputs=[output],
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def if_input_generator(session):
    cond_input = session.get_inputs()[0]
    return {cond_input.name: np.array(True, dtype=bool)}  # ou False pour tester l'autre branche

SpecialModelBuilders["If"] = if_model_builder
SpecialInputGenerators["If"] = if_input_generator
