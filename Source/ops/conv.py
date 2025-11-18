# ops/conv.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def conv_model_builder(op_type, cfg=None):
    # Définir les entrées
    input_tensor = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])  # NCHW
    weight_tensor = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT, [6, 3, 5, 5])   # MCHW
    bias_tensor = onnx.helper.make_tensor_value_info("B", TensorProto.FLOAT, [6])             # M
    output_tensor = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Définir le nœud Conv
    node = onnx.helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        kernel_shape=[5, 5],
        strides=[1, 1],
        pads=[2, 2, 2, 2],  # padding symétrique
        dilations=[1, 1],
        group=1
    )

    # Créer le graphe
    graph = onnx.helper.make_graph(
        [node],
        "conv_graph",
        [input_tensor, weight_tensor, bias_tensor],
        [output_tensor]
    )

    # Créer le modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Conv"] = conv_model_builder

def conv_input_generator(session):
    input_info, weight_info, bias_info = session.get_inputs()
    X = np.random.rand(1, 3, 32, 32).astype(np.float32)
    W = np.random.randn(6, 3, 5, 5).astype(np.float32) * 0.1
    B = np.random.randn(6).astype(np.float32) * 0.01
    return {
        input_info.name: X,
        weight_info.name: W,
        bias_info.name: B
    }

SpecialInputGenerators["Conv"] = conv_input_generator
