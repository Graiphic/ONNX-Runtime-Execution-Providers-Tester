# ops/compress.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def compress_model_builder(op_type, cfg=None):
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [4, 4])
    cond = onnx.helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, [4])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "Compress",
        inputs=["data", "condition"],
        outputs=["output"],
        axis=0  # tu peux enlever l’argument pour aplatir avant compress
    )

    graph = onnx.helper.make_graph(
        [node],
        "compress_graph",
        [data, cond],
        [output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

SpecialModelBuilders["Compress"] = compress_model_builder

def compress_input_generator(session):
    data_info, cond_info = session.get_inputs()
    shape = [d or 1 for d in data_info.shape]
    axis_len = shape[0]  # on applique compress sur l'axe 0 ici

    data = np.random.rand(*shape).astype(np.float32)
    condition = np.random.choice([True, False], size=(axis_len,))
    return {
        data_info.name: data,
        cond_info.name: condition
    }

SpecialInputGenerators["Compress"] = compress_input_generator
