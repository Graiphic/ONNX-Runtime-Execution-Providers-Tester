# ops/microsoft_mulinteger.py

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def mulinteger_model_builder(op_type, cfg=None):
    """
    Construit un modèle ONNX minimal pour com.microsoft::MulInteger,
    en créant deux constantes pour A_zero_point et B_zero_point (valeurs 0) 
    afin que l'op MulInteger reçoive bien des tenseurs au lieu de chaînes vides.
    """

    # 1) Déclaration des value_info pour A (UINT8 [2,3]), B (UINT8 [3]) et la sortie C (INT32 [2,3])
    A = onnx.helper.make_tensor_value_info("A", TensorProto.UINT8, [2, 3])
    B = onnx.helper.make_tensor_value_info("B", TensorProto.UINT8, [3])
    C = onnx.helper.make_tensor_value_info("C", TensorProto.INT32, [2, 3])

    # 2) Création de deux nœuds Constant pour A_zero_point et B_zero_point (valeur 0)
    #    Ils produisent chacun un scalaire UINT8 = 0
    const_A_zp = onnx.helper.make_tensor(
        name="A_zero_point_tensor",
        data_type=TensorProto.UINT8,
        dims=[],             # tenseur scalaire
        vals=[0]
    )
    node_const_A_zp = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["A_zero_point"],
        value=const_A_zp
    )

    const_B_zp = onnx.helper.make_tensor(
        name="B_zero_point_tensor",
        data_type=TensorProto.UINT8,
        dims=[],             # tenseur scalaire
        vals=[0]
    )
    node_const_B_zp = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["B_zero_point"],
        value=const_B_zp
    )

    # 3) Création du nœud MulInteger qui attend 4 entrées : A, A_zero_point, B, B_zero_point
    node_mul = onnx.helper.make_node(
        "MulInteger",
        inputs=["A", "A_zero_point", "B", "B_zero_point"],
        outputs=["C"],
        domain="com.microsoft"
    )

    # 4) Assemblage du graphe : on n’inclut que A et B en tant qu’inputs externes
    graph = onnx.helper.make_graph(
        [node_const_A_zp, node_const_B_zp, node_mul],
        "MulIntegerGraph",
        [A, B],
        [C]
    )

    # 5) Création du modèle ONNX avec import de l’opset standard (pour Constant) et com.microsoft v1 (pour MulInteger)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION),     # ONNX standard
            onnx.helper.make_operatorsetid("com.microsoft", 1)           # MulInteger dans com.microsoft
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    # 6) Vérification du modèle
    onnx.checker.check_model(model)
    return model

def mulinteger_input_generator(session):
    """
    Génère A et B en UINT8 :
      - A : shape [2,3]
      - B : shape [3]
    Les zero_points sont fournis en interne par les Constant du modèle.
    """
    A = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
    B = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    return {"A": A, "B": B}

# Enregistrement dans les dictionnaires globaux pour OpTest
SpecialModelBuilders["com.microsoft.MulInteger"] = mulinteger_model_builder
SpecialInputGenerators["com.microsoft.MulInteger"] = mulinteger_input_generator
