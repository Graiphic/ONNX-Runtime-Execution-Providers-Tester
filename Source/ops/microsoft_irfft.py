# ops/microsoft_irfft.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def irfft_model_builder(op_type, cfg=None):
    # Paramètres du signal
    n = 8
    onesided_length = n // 2 + 1

    # Entrée: (batch_size, onesided_length, 2)
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, onesided_length, 2])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, n])

    node = onnx.helper.make_node(
        "Irfft",
        inputs=["X"],
        outputs=["Y"],
        domain="com.microsoft",
        normalized=0,
        onesided=1,
        signal_ndim=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "IrfftGraph",
        [X],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def irfft_input_generator(session):
    # Simule un spectre RFFT onesided avec valeurs (Re, Im)
    n = 8
    onesided_length = n // 2 + 1
    X = np.random.randn(1, onesided_length, 2).astype(np.float32)
    return {"X": X}

SpecialModelBuilders["com.microsoft.Irfft"] = irfft_model_builder
SpecialInputGenerators["com.microsoft.Irfft"] = irfft_input_generator
