# ops/constantofshape.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def constantofshape_model_builder(op_type, cfg=None):
    shape_input = onnx.helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    output = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    value_tensor = onnx.helper.make_tensor(
        name="value",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[3.14]  # constante de remplissage
    )

    node = onnx.helper.make_node(
        "ConstantOfShape",
        inputs=["shape"],
        outputs=["Y"],
        value=value_tensor
    )

    graph = onnx.helper.make_graph(
        [node],
        "constantofshape_graph",
        [shape_input],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["ConstantOfShape"] = constantofshape_model_builder

def constantofshape_input_generator(session):
    shape_info = session.get_inputs()[0]
    return {shape_info.name: np.array([2, 3], dtype=np.int64)}

SpecialInputGenerators["ConstantOfShape"] = constantofshape_input_generator
