# ops/mod.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def mod_model_builder(op_type, cfg=None):
    inp1 = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [2, 2])
    inp2 = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [2, 2])
    out  = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    # fmod=1 pour forcer le mode flottant
    node = onnx.helper.make_node(op_type, ["A", "B"], ["Y"], fmod=1)

    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [inp1, inp2], [out])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)])
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def mod_input_generator(session):
    input_infos = session.get_inputs()
    shape = [d or 1 for d in input_infos[0].shape]
    A = np.random.uniform(1, 10, size=shape).astype(np.float32)
    B = np.random.uniform(1, 10, size=shape).astype(np.float32)
    return {input_infos[0].name: A, input_infos[1].name: B}

SpecialModelBuilders["Mod"] = mod_model_builder
SpecialInputGenerators["Mod"] = mod_input_generator
