# ops/melweightmatrix.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def melweightmatrix_model_builder(op_type, cfg=None):
    inputs = [
        onnx.helper.make_tensor_value_info("num_mel_bins", TensorProto.INT64, []),
        onnx.helper.make_tensor_value_info("dft_length", TensorProto.INT64, []),
        onnx.helper.make_tensor_value_info("sample_rate", TensorProto.INT64, []),
        onnx.helper.make_tensor_value_info("lower_edge_hertz", TensorProto.FLOAT, []),
        onnx.helper.make_tensor_value_info("upper_edge_hertz", TensorProto.FLOAT, [])
    ]
    output = onnx.helper.make_tensor_value_info("mel_weight_matrix", TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "MelWeightMatrix",
        inputs=["num_mel_bins", "dft_length", "sample_rate", "lower_edge_hertz", "upper_edge_hertz"],
        outputs=["mel_weight_matrix"]
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="melweightmatrix_graph",
        inputs=inputs,
        outputs=[output]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def melweightmatrix_input_generator(session):
    return {
        "num_mel_bins": np.array(40, dtype=np.int64),
        "dft_length": np.array(512, dtype=np.int64),
        "sample_rate": np.array(16000, dtype=np.int64),
        "lower_edge_hertz": np.array(20.0, dtype=np.float32),
        "upper_edge_hertz": np.array(8000.0, dtype=np.float32)
    }

SpecialModelBuilders["MelWeightMatrix"] = melweightmatrix_model_builder
SpecialInputGenerators["MelWeightMatrix"] = melweightmatrix_input_generator
