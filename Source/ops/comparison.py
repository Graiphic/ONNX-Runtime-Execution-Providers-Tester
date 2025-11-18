# ops/comparison.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

# Liste des opérateurs de comparaison pris en charge
COMPARISON_OPS = ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"]

def comparison_model_builder(op_type, cfg=None):
    # Définition des entrées et de la sortie
    inp1 = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 2])
    inp2 = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [2, 2])
    out  = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.BOOL, None)

    # Création du nœud de l'opérateur
    node = onnx.helper.make_node(op_type, inputs=["A", "B"], outputs=["Y"])

    # Construction du graphe et du modèle
    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [inp1, inp2], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def comparison_input_generator(session):
    # Génération d'entrées aléatoires
    inputs = session.get_inputs()
    shape = [d if isinstance(d, int) else 1 for d in inputs[0].shape]
    A = np.random.rand(*shape).astype(np.float32)
    B = np.random.rand(*shape).astype(np.float32)
    return {inputs[0].name: A, inputs[1].name: B}

# Enregistrement des builders et générateurs pour chaque opérateur
for op_type in COMPARISON_OPS:
    SpecialModelBuilders[op_type] = comparison_model_builder
    SpecialInputGenerators[op_type] = comparison_input_generator
