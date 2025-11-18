# ops/meanvariancenormalization.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def mvn_model_builder(op_type, cfg=None):
    x = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "MeanVarianceNormalization",
        inputs=["X"],
        outputs=["Y"],
        axes=[2]  # normalisation sur l'axe largeur, par exemple
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="mvn_graph",
        inputs=[x],
        outputs=[y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def mvn_input_generator(session):
    input_info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    x = np.random.rand(*shape).astype(np.float32)
    return {input_info.name: x}

SpecialModelBuilders["MeanVarianceNormalization"] = mvn_model_builder
SpecialInputGenerators["MeanVarianceNormalization"] = mvn_input_generator
