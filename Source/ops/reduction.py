# ops/reduction.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

# Liste des opérateurs de réduction pris en charge
REDUCTION_OPS = [
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"
]

def reduction_model_builder(op_type, cfg=None):
    # Entrée principale
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 4, 5])
    # Entrée facultative 'axes'
    axes = onnx.helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])
    # Sortie
    out = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    # Attributs par défaut pour opset 22
    node = onnx.helper.make_node(
        op_type,
        inputs=["data", "axes"],
        outputs=["Y"],
        keepdims=1,
        noop_with_empty_axes=0
    )

    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [data, axes], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def reduction_input_generator(session):
    data_info, axes_info = session.get_inputs()
    shape = [d if isinstance(d, int) else 1 for d in data_info.shape]
    data = np.random.rand(*shape).astype(np.float32)
    axes = np.array([1], dtype=np.int64)  # Réduction sur l'axe 1
    return {data_info.name: data, axes_info.name: axes}

# Enregistrement
for op_type in REDUCTION_OPS:
    SpecialModelBuilders[op_type] = reduction_model_builder
    SpecialInputGenerators[op_type] = reduction_input_generator
