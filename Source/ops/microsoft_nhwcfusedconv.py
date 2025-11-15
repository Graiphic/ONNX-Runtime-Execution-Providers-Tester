# ops/microsoft_nhwcfusedconv.py

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def nhwcfusedconv_model_builder(op_type, cfg=None):
    """
    On définit X en float32, on cast explicitement vers float16,
    puis on appelle NhwcFusedConv en float16. Ainsi ONNXRuntime 
    n'insérera pas un Cast automatique incorrect.
    """

    # --- 1) Paramètres : on garde C_in et C_out multiples de 4 pour le kernel GPU ---
    N, H, Wd, C_in, C_out, k = 1, 5, 5, 4, 4, 3

    # --- 2) Déclaration des shapes et des types des entrées/sorties ---
    #    - X en float32, shape NHWC
    X = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [N, H, Wd, C_in])
    #    - W en float16, shape OIHW
    W = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT16, [C_out, C_in, k, k])
    #    - Y (la sortie) en float16, shape NHWC
    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [N, H, Wd, C_out])

    nodes = []

    # --- 3) Premier nœud : Cast X(float32) → X_fp16(float16) ---
    #    ONNXRuntime verra que X est float32, prendra en charge ce Cast,
    #    et produira X_fp16 que NhwcFusedConv pourra consommer sans insertion automatique.
    nodes.append(
        onnx.helper.make_node(
            "Cast",
            inputs=["X"],
            outputs=["X_fp16"],
            to=TensorProto.FLOAT16
        )
    )

    # --- 4) Nœud NhwcFusedConv (en float16) ---
    nodes.append(
        onnx.helper.make_node(
            "NhwcFusedConv",
            inputs=["X_fp16", "W"],
            outputs=["Y"],
            domain="com.microsoft",
            kernel_shape=[k, k],
            pads=[1, 1, 1, 1],  # top, left, bottom, right
            strides=[1, 1],
            group=1
            # si besoin, ajouter activation="Relu" ou bias B via inputs supplémentaires
        )
    )

    # --- 5) On assemble le graphe principal ---
    graph = onnx.helper.make_graph(
        nodes,
        "NhwcFusedConvGraph",
        [X, W],
        [Y]
    )

    # --- 6) Imports d’opsets : ONNX standard + com.microsoft v1 ---
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION),      # standard ONNX (pour Cast)
            onnx.helper.make_operatorsetid("com.microsoft", 1)            # pour NhwcFusedConv
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    # --- 7) Vérification rapide du modèle ---
    onnx.checker.check_model(model)
    return model

def nhwcfusedconv_input_generator(session):
    """
    Génère :
      - X en float32, shape [1,5,5,4]
      - W en float16, shape [4,4,3,3]
    """
    N, H, Wd, C_in, C_out, k = 1, 5, 5, 4, 4, 3
    # X en float32
    X = np.random.randn(N, H, Wd, C_in).astype(np.float32)
    # W en float16
    W = np.random.randn(C_out, C_in, k, k).astype(np.float16)
    return {"X": X, "W": W}

# Enregistrement pour OpTest
SpecialModelBuilders["com.microsoft.NhwcFusedConv"]   = nhwcfusedconv_model_builder
SpecialInputGenerators["com.microsoft.NhwcFusedConv"] = nhwcfusedconv_input_generator
