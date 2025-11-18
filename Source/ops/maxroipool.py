# ops/maxroipool.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def maxroipool_model_builder(op_type, cfg=None):
    # Définition des entrées
    X = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
    rois = onnx.helper.make_tensor_value_info("rois", TensorProto.FLOAT, [None, 5])
    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Création du nœud MaxRoiPool
    node = onnx.helper.make_node(
        "MaxRoiPool",
        inputs=["X", "rois"],
        outputs=["Y"],
        pooled_shape=[7, 7],
        spatial_scale=1.0
    )

    # Construction du graphe
    graph = onnx.helper.make_graph(
        nodes=[node],
        name="maxroipool_graph",
        inputs=[X, rois],
        outputs=[Y]
    )

    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def maxroipool_input_generator(session):
    # Récupération des informations sur les entrées
    input_info = {inp.name: inp for inp in session.get_inputs()}
    X_shape = [d if isinstance(d, int) else 1 for d in input_info["X"].shape]

    # Génération de données aléatoires pour X
    X_data = np.random.rand(*X_shape).astype(np.float32)

    # Définition de RoIs : [batch_index, x1, y1, x2, y2]
    rois_data = np.array([
        [0, 10.0, 10.0, 30.0, 30.0],
        [0, 20.0, 20.0, 50.0, 50.0]
    ], dtype=np.float32)

    return {
        "X": X_data,
        "rois": rois_data
    }

SpecialModelBuilders["MaxRoiPool"] = maxroipool_model_builder
SpecialInputGenerators["MaxRoiPool"] = maxroipool_input_generator
