# ops/deformconv.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def deformconv_model_builder(op_type, cfg=None):
    # Définition des entrées
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
    w = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, [6, 3, 3, 3])
    offset = onnx.helper.make_tensor_value_info("offset", TensorProto.FLOAT, [1, 18, 32, 32])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Création du nœud DeformConv avec les attributs requis
    node = onnx.helper.make_node(
        "DeformConv",
        inputs=["X", "W", "offset"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
        offset_group=1
    )

    # Création du graphe
    graph = onnx.helper.make_graph(
        [node],
        "deformconv_graph",
        [x, w, offset],
        [y]
    )
    #print("\nle sang\n")
    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["DeformConv"] = deformconv_model_builder

def deformconv_input_generator(session):
    x_info, w_info, offset_info = session.get_inputs()
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    w = np.random.randn(6, 3, 3, 3).astype(np.float32) * 0.1
    offset = np.random.randn(1, 18, 32, 32).astype(np.float32) * 0.1
    return {
        x_info.name: x,
        w_info.name: w,
        offset_info.name: offset
    }

SpecialInputGenerators["DeformConv"] = deformconv_input_generator
