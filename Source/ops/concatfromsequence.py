# ops/concatfromsequence.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def concatfromsequence_model_builder(op_type, cfg=None):
    # Définir l'entrée de type séquence de tenseurs
    input_sequence = onnx.helper.make_tensor_sequence_value_info("input_sequence", TensorProto.FLOAT, None)
    output = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, None)

    # Définir le nœud ConcatFromSequence
    node = onnx.helper.make_node(
        "ConcatFromSequence",
        inputs=["input_sequence"],
        outputs=["output"],
        axis=0,       # Axe de concaténation
        new_axis=0    # Si 1, ajoute un nouvel axe avant la concaténation
    )

    # Créer le graphe
    graph = onnx.helper.make_graph(
        [node],
        "concatfromsequence_graph",
        [input_sequence],
        [output]
    )

    # Créer le modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["ConcatFromSequence"] = concatfromsequence_model_builder

def concatfromsequence_input_generator(session):
    input_info = session.get_inputs()[0]
    # Générer une séquence de 3 tenseurs de forme (2, 3)
    tensors = [np.random.rand(2, 3).astype(np.float32) for _ in range(3)]
    return {input_info.name: tensors}

SpecialInputGenerators["ConcatFromSequence"] = concatfromsequence_input_generator
