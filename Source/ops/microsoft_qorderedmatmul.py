# ops/microsoft_qorderedmatmul.py

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qorderedmatmul_model_builder(op_type, cfg=None):
    # ─── 1) Paramètres et attributs ────────────────────────────────────────
    # A en row-major, B sera lu en column-major (ORDER_COL), Y en row-major.
    order_A = 1  # ORDER_ROW
    order_B = 0  # ORDER_COL  (lécture CUDA en column-major)
    order_Y = 1  # ORDER_ROW

    BATCH = 1
    M = 4   # nombre de lignes de A
    K = 8   # dimension intermédiaire (A.shape[2] et B.shape[0])
    N = 4   # nombre de colonnes de Y

    # ─── 2) Définition de l’input A ────────────────────────────────────────
    # A doit être [batch, M, K] = [1, 4, 8] en INT8 row-major
    def make_input(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    A = make_input("A", TensorProto.INT8, [BATCH, M, K])
    # On ne déclare PAS B ici en tant qu'input, car on créera un initializer pour B.
    inputs_info = [A]

    # ─── 3) Création des initializers / constantes ─────────────────────────
    inits = []
    def add_const(name, array, dtype):
        tensor = onnx.helper.make_tensor(
            name=name,
            data_type=dtype,
            dims=array.shape,
            vals=array.flatten().tolist()
        )
        inits.append(tensor)

    # 3.1 – scale_A : scalaire float32
    add_const("scale_A", np.array(0.02, dtype=np.float32), TensorProto.FLOAT)

    # 3.2 – B : on crée B en [K, N] = [8, 4] pour pouvoir garder order_B = ORDER_COL
    #       (ce tableau INT8 est rangé en row-major, CUDA lira ensuite en column-major)
    B_data = np.random.randint(-128, 127, size=(K, N), dtype=np.int8)
    add_const("B", B_data, TensorProto.INT8)

    # 3.3 – scale_B : per-column quant, vecteur de longueur N = 4
    scale_B = np.ones((N,), dtype=np.float32)
    add_const("scale_B", scale_B, TensorProto.FLOAT)

    # 3.4 – scale_Y : scalaire float32 (output)
    add_const("scale_Y", np.array(0.04, dtype=np.float32), TensorProto.FLOAT)

    # ─── 4) Liste EXACTE des noms d’inputs pour le node ────────────────────
    # Même si B est un initializer, il faut l’inclure ici pour que ONNX shape inference
    # sache que “B” existe dans le graphe (avec shape [8,4]).
    input_names = ["A", "scale_A", "B", "scale_B", "scale_Y"]

    # ─── 5) Définition de la sortie Y ──────────────────────────────────────
    # Y sera un int8 tensor shape [batch, M, N] = [1, 4, 4]
    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.INT8, [BATCH, M, N])

    # ─── 6) Création du node QOrderedMatMul ────────────────────────────────
    node = onnx.helper.make_node(
        "QOrderedMatMul",
        inputs=input_names,
        outputs=["Y"],
        domain="com.microsoft",
        order_A=order_A,
        order_B=order_B,
        order_Y=order_Y
    )

    # ─── 7) Assemblage du graphe et création du modèle ONNX ───────────────
    graph = onnx.helper.make_graph(
        [node],
        "QOrderedMatMulGraph",
        inputs_info,
        [Y],
        initializer=inits
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    # Vérification de cohérence du modèle (shape inference doit valider [1,4,8] × [8,4] → [1,4,4])
    onnx.checker.check_model(model)
    return model


def qorderedmatmul_input_generator(session):
    """
    Génère uniquement A (INT8) de shape [batch, M, K]. 
    scale_A, B, scale_B, scale_Y sont déjà dans les initializers.
    """
    # On récupère la shape attendue pour A
    info_A = next(inp for inp in session.get_inputs() if inp.name == "A")
    shape_A = [d if isinstance(d, int) else 1 for d in info_A.shape]  # [1,4,8]
    data_A = np.random.randint(-128, 127, size=shape_A, dtype=np.int8)
    return {"A": data_A}


# Enregistrement pour OpTest
SpecialModelBuilders["com.microsoft.QOrderedMatMul"]   = qorderedmatmul_model_builder
SpecialInputGenerators["com.microsoft.QOrderedMatMul"] = qorderedmatmul_input_generator
