# ops/hannwindow.py
import numpy as np
import onnx
from onnx import helper, TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def hannwindow_model_builder(op_type, cfg=None):
    # 1. Déclaration de l'entrée “size” comme scalaire INT64
    size_input = helper.make_tensor_value_info("size", TensorProto.INT64, [])

    # 2. Déclaration de la sortie
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # 3. Création du nœud HannWindow
    node = helper.make_node(
        "HannWindow",
        inputs=["size"],
        outputs=["Y"],
        periodic=1,             # periodic=True
        output_datatype=TensorProto.FLOAT
    )

    # 4. Assemblage du graphe sans initializer
    graph = helper.make_graph(
        nodes=[node],
        name="hannwindow_graph",
        inputs=[size_input],    # size devient un input
        outputs=[output]
    )

    # 5. Création du modèle ONNX
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["HannWindow"] = hannwindow_model_builder


def hannwindow_input_generator(session):
    # On fournit maintenant “size” en entrée au lieu d'un initializer
    inp = session.get_inputs()[0]   # récupère size
    return {
        inp.name: np.array(10, dtype=np.int64)  # ici taille=10
    }

SpecialInputGenerators["HannWindow"] = hannwindow_input_generator
