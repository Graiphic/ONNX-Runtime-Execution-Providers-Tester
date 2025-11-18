# ops/dynamicquantizelstm.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def dynamicquantizelstm_model_builder(op_type, cfg=None):
    seq_len = 3
    batch_size = 2
    input_size = 4
    hidden_size = 5
    num_directions = 1
    w_zp_dtype = onnx.TensorProto.INT8
    r_zp_dtype = onnx.TensorProto.INT8

    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [seq_len, batch_size, input_size])
    W = onnx.helper.make_tensor_value_info("W", onnx.TensorProto.INT8, [num_directions, input_size, 4 * hidden_size])
    R = onnx.helper.make_tensor_value_info("R", onnx.TensorProto.INT8, [num_directions, hidden_size, 4 * hidden_size])
    W_scale = onnx.helper.make_tensor_value_info("W_scale", onnx.TensorProto.FLOAT, [num_directions])
    W_zp = onnx.helper.make_tensor_value_info("W_zero_point", w_zp_dtype, [num_directions])
    R_scale = onnx.helper.make_tensor_value_info("R_scale", onnx.TensorProto.FLOAT, [num_directions])
    R_zp = onnx.helper.make_tensor_value_info("R_zero_point", r_zp_dtype, [num_directions])

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    Y_h = onnx.helper.make_tensor_value_info("Y_h", onnx.TensorProto.FLOAT, None)
    Y_c = onnx.helper.make_tensor_value_info("Y_c", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DynamicQuantizeLSTM",
        inputs=["X", "W", "R", "", "", "", "", "", "W_scale", "W_zero_point", "R_scale", "R_zero_point"],
        outputs=["Y", "Y_h", "Y_c"],
        domain="com.microsoft",
        hidden_size=hidden_size,
        direction="forward"
    )

    graph = onnx.helper.make_graph(
        [node],
        "DynamicQuantizeLSTMGraph",
        [X, W, R, W_scale, W_zp, R_scale, R_zp],
        [Y, Y_h, Y_c]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def dynamicquantizelstm_input_generator(session):
    seq_len = 3
    batch_size = 2
    input_size = 4
    hidden_size = 5
    num_directions = 1

    X = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
    W = np.random.randint(-128, 127, size=(num_directions, input_size, 4 * hidden_size), dtype=np.int8)
    R = np.random.randint(-128, 127, size=(num_directions, hidden_size, 4 * hidden_size), dtype=np.int8)
    W_scale = np.random.rand(num_directions).astype(np.float32)
    W_zp = np.random.randint(-128, 127, size=num_directions, dtype=np.int8)
    R_scale = np.random.rand(num_directions).astype(np.float32)
    R_zp = np.random.randint(-128, 127, size=num_directions, dtype=np.int8)

    return {
        "X": X,
        "W": W,
        "R": R,
        "W_scale": W_scale,
        "W_zero_point": W_zp,
        "R_scale": R_scale,
        "R_zero_point": R_zp
    }

SpecialModelBuilders["com.microsoft.DynamicQuantizeLSTM"] = dynamicquantizelstm_model_builder
SpecialInputGenerators["com.microsoft.DynamicQuantizeLSTM"] = dynamicquantizelstm_input_generator
