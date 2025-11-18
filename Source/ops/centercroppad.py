# ops/centercroppad.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def centercroppad_model_builder(op_type, cfg=None):
    # Définir les entrées
    input_data = onnx.helper.make_tensor_value_info("input_data", onnx.TensorProto.FLOAT, [1, 3, 28, 28])
    shape = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [2])
    output_data = onnx.helper.make_tensor_value_info("output_data", onnx.TensorProto.FLOAT, None)

    # Définir le nœud CenterCropPad
    node = onnx.helper.make_node(
        "CenterCropPad",
        inputs=["input_data", "shape"],
        outputs=["output_data"],
        axes=[2, 3]  # Appliquer sur les dimensions H et W
    )

    # Créer le graphe
    graph = onnx.helper.make_graph(
        [node],
        "centercroppad_graph",
        [input_data, shape],
        [output_data]
    )

    # Créer le modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["CenterCropPad"] = centercroppad_model_builder

def centercroppad_input_generator(session):
    input_info, shape_info = session.get_inputs()
    input_shape = [d or 1 for d in input_info.shape]
    input_data = np.random.rand(*input_shape).astype(np.float32)
    target_shape = np.array([20, 20], dtype=np.int64)  # Dimensions cibles pour H et W
    return {
        input_info.name: input_data,
        shape_info.name: target_shape
    }

SpecialInputGenerators["CenterCropPad"] = centercroppad_input_generator
