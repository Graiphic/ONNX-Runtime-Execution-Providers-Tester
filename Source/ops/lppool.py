# ops/lppool.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def lppool_model_builder(op_type, cfg=None):
    # Définition de l'entrée et de la sortie
    inp = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    out = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Création du nœud LpPool
    node = onnx.helper.make_node(
        "LpPool",
        inputs=["X"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        p=2  # Norme L2
    )

    # Construction du graphe
    graph = onnx.helper.make_graph([node], "lppool_graph", [inp], [out])

    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def lppool_input_generator(session):
    # Génération d'une entrée aléatoire
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {input_info.name: x}

# Enregistrement des builders et générateurs
SpecialModelBuilders["LpPool"] = lppool_model_builder
SpecialInputGenerators["LpPool"] = lppool_input_generator
