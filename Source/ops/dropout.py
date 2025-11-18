# ops/dropout.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def dropout_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3, 4])
    ratio = onnx.helper.make_tensor_value_info("ratio", onnx.TensorProto.FLOAT, [])
    training_mode = onnx.helper.make_tensor_value_info("training", onnx.TensorProto.BOOL, [])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Dropout",
        inputs=["data", "ratio", "training"],
        outputs=["output"]
    )

    graph = onnx.helper.make_graph([node], "dropout_graph", [data, ratio, training_mode], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def dropout_input_generator(session):
    data_info, ratio_info, training_info = session.get_inputs()
    data = np.random.rand(3, 4).astype(np.float32)
    ratio = np.array(0.3, dtype=np.float32)
    training = np.array(True, dtype=bool)
    return {
        data_info.name: data,
        ratio_info.name: ratio,
        training_info.name: training
    }

SpecialModelBuilders["Dropout"] = dropout_model_builder
SpecialInputGenerators["Dropout"] = dropout_input_generator
