# ops/fusedmatmulactivation.py

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def fusedmatmulactivation_model_builder(op_type, cfg=None):
    """
    Construit un modèle ONNX contenant un seul nœud FusedMatMulActivation
    (multiplication de A [2×4] par B [4×3] + Relu).
    """
    # Vérifier que les constantes d’opset et d’IR sont bien définies
    # (sinon fournir des valeurs par défaut raisonnables).
    try:
        onnx_opset = int(ONNX_OPSET_VERSION)
    except Exception:
        onnx_opset = 13  # valeur de secours
    try:
        onnx_ir = int(ONNX_RUNTIME_IR_VERSION)
    except Exception:
        onnx_ir = 7      # valeur de secours

    # 1) Définition des value_info pour A (2×4) et B (4×3)
    A_info = onnx.helper.make_tensor_value_info(
        "A", TensorProto.FLOAT, [2, 4]
    )
    B_info = onnx.helper.make_tensor_value_info(
        "B", TensorProto.FLOAT, [4, 3]
    )
    # 2) Définition explicite de la sortie Y (2×3)
    Y_info = onnx.helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [2, 3]
    )

    # 3) Création du nœud FusedMatMulActivation
    node = onnx.helper.make_node(
        "FusedMatMulActivation",
        inputs=["A", "B"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=1.0,
        transA=0,
        transB=0,
        transBatchA=0,
        transBatchB=0,
        activation="Relu"
    )

    # 4) Assemblage du graphe
    graph = onnx.helper.make_graph(
        [node],
        "FusedMatMulActivationGraph",
        [A_info, B_info],
        [Y_info]
    )

    # 5) Création du modèle avec import des opérateurs ONNX standard et com.microsoft v1
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", onnx_opset),
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        ],
        ir_version=onnx_ir
    )

    # Validation rapide du modèle : si ça lève une Exception, vous verrez le message d’erreur
    onnx.checker.check_model(model)
    return model

def fusedmatmulactivation_input_generator(session):
    """
    Génère deux matrices aléatoires (float32) de forme adéquate pour A et B.
    La session n’est pas utilisée ici, mais la signature doit matcher l’appel de OpTest.
    """
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(4, 3).astype(np.float32)
    return {"A": A, "B": B}

# Enregistrement dans les dictionnaires globaux pour OpTest
SpecialModelBuilders["com.microsoft.FusedMatMulActivation"] = fusedmatmulactivation_model_builder
SpecialInputGenerators["com.microsoft.FusedMatMulActivation"] = fusedmatmulactivation_input_generator
