# ops/microsoft_qorderedgelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qorderedgelu_model_builder(op_type, cfg=None):
    order = 1  # ORDER_ROW
    shape = [2, 16]

    # Inputs dynamiques (X uniquement)
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.INT8, shape)
    inputs_info = [X]

    # Initializers : scales
    inits = []
    def add_const(name, val):
        t = onnx.helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[val]
        )
        inits.append(t)

    add_const("scale_X", 0.02)
    add_const("scale_Y", 0.02)

    input_names = ["X", "scale_X", "scale_Y"]
    output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.INT8, None)

    node = onnx.helper.make_node(
        "QOrderedGelu",
        inputs=input_names,
        outputs=["Y"],
        domain="com.microsoft",
        order_X=order,
        order_Y=order
    )

    graph = onnx.helper.make_graph(
        [node],
        "QOrderedGeluGraph",
        inputs_info,
        [output],
        initializer=inits
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def qorderedgelu_input_generator(session):
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    return {inp.name: np.random.randint(-128, 127, size=shape, dtype=np.int8)}

SpecialModelBuilders["com.microsoft.QOrderedGelu"] = qorderedgelu_model_builder
SpecialInputGenerators["com.microsoft.QOrderedGelu"] = qorderedgelu_input_generator
