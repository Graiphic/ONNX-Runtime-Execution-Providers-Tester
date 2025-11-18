# ops/constant.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def constant_model_builder(op_type, cfg=None):
    # Création d'une constante 2x2 float32
    const_tensor = onnx.helper.make_tensor(
        name="const_tensor",
        data_type=TensorProto.FLOAT,
        dims=[2, 2],
        vals=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).flatten().tolist()
    )

    node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["Y"],
        value=const_tensor
    )

    output = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    graph = onnx.helper.make_graph(
        [node],
        "constant_graph",
        [],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Constant"] = constant_model_builder

def constant_input_generator(session):
    # Aucun input nécessaire pour Constant
    return {}

SpecialInputGenerators["Constant"] = constant_input_generator
