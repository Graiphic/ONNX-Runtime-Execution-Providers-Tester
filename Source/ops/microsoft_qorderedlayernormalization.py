# ops/microsoft_qorderedlayernormalization.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qorderedlayernorm_model_builder(op_type, cfg=None):
    shape = [1, 4, 8]
    axis = 2
    epsilon = 1e-5
    order = 1  # ORDER_ROW

    # Entrée dynamique : X
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.INT8, shape)
    inputs_info = [X]

    # Initializers
    inits = []
    def add_const(name, array, dtype):
        t = onnx.helper.make_tensor(
            name=name,
            data_type=dtype,
            dims=array.shape,
            vals=array.flatten().tolist()
        )
        inits.append(t)

    add_const("scale_X", np.array(0.02, dtype=np.float32), onnx.TensorProto.FLOAT)
    add_const("scale_Y", np.array(0.02, dtype=np.float32), onnx.TensorProto.FLOAT)
    add_const("scale", np.ones(shape[-1], dtype=np.float32), onnx.TensorProto.FLOAT)  # gamma

    input_names = ["X", "scale_X", "scale", "", "scale_Y"]  # "" pour B (bias) optionnel

    output = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.INT8, None)

    node = onnx.helper.make_node(
        "QOrderedLayerNormalization",
        inputs=input_names,
        outputs=["Y"],
        domain="com.microsoft",
        axis=axis,
        epsilon=epsilon,
        order_X=order,
        order_Y=order
    )

    graph = onnx.helper.make_graph(
        [node],
        "QOrderedLayerNormalizationGraph",
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

def qorderedlayernorm_input_generator(session):
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    return {inp.name: np.random.randint(-128, 127, size=shape, dtype=np.int8)}

SpecialModelBuilders["com.microsoft.QOrderedLayerNormalization"] = qorderedlayernorm_model_builder
SpecialInputGenerators["com.microsoft.QOrderedLayerNormalization"] = qorderedlayernorm_input_generator
