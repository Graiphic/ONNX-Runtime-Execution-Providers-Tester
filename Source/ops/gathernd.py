# ops/gathernd.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def gathernd_model_builder(op_type, cfg=None):
    # Définition des entrées
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 2, 2])
    indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2, 2])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    # Création du nœud GatherND
    node = onnx.helper.make_node(
        "GatherND",
        inputs=["data", "indices"],
        outputs=["output"],
        batch_dims=0  # Attribut optionnel
    )

    # Construction du graphe et du modèle
    graph = onnx.helper.make_graph([node], "gathernd_graph", [data, indices], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def gathernd_input_generator(session):
    # Récupération des informations sur les entrées
    data_info, indices_info = session.get_inputs()
    # Génération de données d'entrée aléatoires
    data = np.random.rand(2, 2, 2).astype(np.float32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return {data_info.name: data, indices_info.name: indices}

# Enregistrement des fonctions dans les dictionnaires
SpecialModelBuilders["GatherND"] = gathernd_model_builder
SpecialInputGenerators["GatherND"] = gathernd_input_generator
