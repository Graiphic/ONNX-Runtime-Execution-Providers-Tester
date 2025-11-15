import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def qorderedattention_model_builder(op_type, cfg=None):
    """
    Modèle ONNX pour com.microsoft::QOrderedAttention corrigé :
    - order_weight passe à 1 car NumPy alloue W en row-major.
    - mask_index est défini sur S-1 (le dernier indice valide), pour éviter l'accès hors bornes.
    """

    # 1) Paramètres
    num_heads    = 2
    order_input  = 1   # On garde ROW major pour l'input
    order_output = 1   # ROW major pour la sortie
    order_weight = 1   # ROW major pour les poids (on ne force plus FORTRAN)
    batch_size, S, H = 1, 4, 8
    head_size    = H // num_heads    # 4
    hidden_size  = num_heads * head_size  # 8

    # Petit utilitaire pour créer un ValueInfoProto
    def make_input(name, dtype, shape):
        return onnx.helper.make_tensor_value_info(name, dtype, shape)

    # 2) Déclaration de l’entrée principale : INT8 [batch_size, S, H]
    inputs_info = [
        make_input("input", TensorProto.INT8, [batch_size, S, H])
    ]

    # 3) Liste d’initializers / constantes
    inits = []
    def add_const(name, array, dtype):
        t = onnx.helper.make_tensor(
            name=name,
            data_type=dtype,
            dims=array.shape,
            vals=array.flatten().tolist()
        )
        inits.append(t)

    # 4) Création des constantes d’échelle
    scalar_names = [
        "scale_input", "scale_Q_gemm", "scale_K_gemm", "scale_V_gemm",
        "scale_Q_weight", "scale_K_weight", "scale_V_weight",
        "scale_QKT_gemm", "scale_QKT_softmax", "scale_values_gemm"
    ]
    for name in scalar_names:
        if name in ("scale_QKT_gemm", "scale_QKT_softmax"):
            val = np.array(1.0, dtype=np.float32)
        else:
            val = np.array(0.02, dtype=np.float32)
        add_const(name, val, TensorProto.FLOAT)

    # 5) Poids (Q, K, V) et biais
    for proj in ["Q", "K", "V"]:
        # Poids INT8 : shape [H, hidden_size] = [8, 8]
        # On reste en row-major, donc order_weight = 1
        W = np.random.randint(-128, 127, size=(H, hidden_size), dtype=np.int8)
        add_const(f"{proj}_weight", W, TensorProto.INT8)

        # Biais FLOAT : shape [hidden_size] = [8]
        biais_proj = np.random.randn(hidden_size).astype(np.float32)
        add_const(f"{proj}_bias", biais_proj, TensorProto.FLOAT)

    # 6) mask_index avec valeur S-1 (indice inclusif)
    mask = np.array([S - 1], dtype=np.int32)  # i.e. [3]
    add_const("mask_index", mask, TensorProto.INT32)

    # 7) Liste des noms d’inputs pour le nœud
    input_names = [
        "input",             # 0
        "scale_input",       # 1
        "scale_Q_gemm",      # 2
        "scale_K_gemm",      # 3
        "scale_V_gemm",      # 4
        "Q_weight",          # 5
        "K_weight",          # 6
        "V_weight",          # 7
        "scale_Q_weight",    # 8
        "scale_K_weight",    # 9
        "scale_V_weight",    # 10
        "Q_bias",            # 11
        "K_bias",            # 12
        "V_bias",            # 13
        "scale_QKT_gemm",    # 14
        "scale_QKT_softmax", # 15
        "scale_values_gemm", # 16
        "mask_index"         # 17
    ]

    # 8) Définition de la sortie : INT8 [batch_size, S, hidden_size]
    output = onnx.helper.make_tensor_value_info(
        "output", TensorProto.INT8, [batch_size, S, hidden_size]
    )

    # 9) Création du node QOrderedAttention
    node = onnx.helper.make_node(
        "QOrderedAttention",
        inputs=input_names,
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        order_input=order_input,
        order_output=order_output,
        order_weight=order_weight,
        unidirectional=1
        # On ne passe pas qkv_hidden_sizes, attention_bias ni past/present
    )

    # 10) Assemblage du graphe
    graph = onnx.helper.make_graph(
        [node],
        "QOrderedAttentionGraph",
        inputs_info,
        [output],
        initializer=inits
    )

    # 11) Création du modèle ONNX
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION),
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    # Vérification
    onnx.checker.check_model(model)
    return model

def qorderedattention_input_generator(session):
    """
    Génère l’entrée 'input' et le 'mask_index' corrigé.
    """
    inputs = session.get_inputs()
    feed = {}

    # Générer aléatoirement l’input INT8 [batch_size, S, H]
    input_info = next(x for x in inputs if x.name == "input")
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    data = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    feed["input"] = data

    # Récupérer et injecter mask_index si présent
    if any(x.name == "mask_index" for x in inputs):
        # On réutilise S-1
        mask_info = next(x for x in inputs if x.name == "mask_index")
        mask_val = np.array([shape[1] - 1], dtype=np.int32)
        feed["mask_index"] = mask_val

    return feed

# Enregistrement pour OpTest
SpecialModelBuilders["com.microsoft.QOrderedAttention"] = qorderedattention_model_builder
SpecialInputGenerators["com.microsoft.QOrderedAttention"] = qorderedattention_input_generator
