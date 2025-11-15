# ops/microsoft_snpe.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def snpe_model_builder(op_type, cfg=None):
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1000])

    node = onnx.helper.make_node(
        "Snpe",
        inputs=["input"],
        outputs=["output"],
        domain="com.microsoft",
        DLC="dummy_dlc_payload",  # Remplacer par le contenu réel du fichier DLC encodé en base64
        notes="Test SNPE operator",
        snpe_version="1.61.0",
        target_device="DSP"
    )

    graph = onnx.helper.make_graph(
        [node],
        "SnpeGraph",
        [input_tensor],
        [output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def snpe_input_generator(session):
    inputs = session.get_inputs()
    feed = {}
    for inp in inputs:
        shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed

SpecialModelBuilders["com.microsoft.Snpe"] = snpe_model_builder
SpecialInputGenerators["com.microsoft.Snpe"] = snpe_input_generator
