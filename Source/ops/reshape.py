# ops/reshape.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def reshape_model_builder(op_type, cfg=None):
    # Définition des entrées et de la sortie
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3, 4])
    shape = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    # Création du nœud Reshape avec l'attribut allowzero
    node = onnx.helper.make_node(
        op_type,
        inputs=["data", "shape"],
        outputs=["output"],
        allowzero=0  # Par défaut, les dimensions avec 0 sont copiées depuis l'entrée
    )

    # Construction du graphe et du modèle
    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [data, shape], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def reshape_input_generator(session):
    # Génération d'un tenseur d'entrée aléatoire et de la nouvelle forme
    data_info, shape_info = session.get_inputs()
    data_shape = [d if isinstance(d, int) else 1 for d in data_info.shape]
    data = np.random.rand(*data_shape).astype(np.float32)

    # Nouvelle forme : [6, 4] pour un total de 24 éléments
    new_shape = np.array([6, 4], dtype=np.int64)
    return {data_info.name: data, shape_info.name: new_shape}

# Enregistrement des builders et générateurs pour l'opérateur Reshape
SpecialModelBuilders["Reshape"] = reshape_model_builder
SpecialInputGenerators["Reshape"] = reshape_input_generator
