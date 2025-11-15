# ops/microsoft_sampling.py

import numpy as np
import onnx
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def sampling_model_builder(op_type, cfg=None):
    """
    Load the Tiny GPT-2 Sampling model from the ONNX file.
    """
    # Replace this path with the location of your .onnx file
    model_path = "tiny_gpt2_sampling.onnx"
    model = onnx.load(model_path)
    # print(model.graph)
    return model

def sampling_input_generator(session):
    """
    Generate the input dictionary identical to the C++ test, namely:
      - input_ids         : int32 [3, 12]
      - max_length        : int32 [1]
      - min_length        : int32 [1]
      - repetition_penalty: float32 [1]
    """
    # input_ids: [3, 12] as in the C++ test
    input_ids = np.array([
         0,   0,   0,   0,   0,  52, 195, 731, 321, 301, 734, 620,
        41, 554,  74, 622, 206, 222,  75, 223, 221, 198, 224, 572,
         0,   0,   0,  52, 328, 219, 328, 206, 288, 227, 896, 328
    ], dtype=np.int32).reshape(3, 12)

    # max_length: [1]
    max_length = np.array([15], dtype=np.int32)

    # min_length: [1]
    min_length = np.array([1], dtype=np.int32)

    # repetition_penalty: [1]
    repetition_penalty = np.array([1.0], dtype=np.float32)

    return {
        "input_ids":          input_ids,
        "max_length":         max_length,
        "min_length":         min_length,
        "repetition_penalty": repetition_penalty
    }

SpecialModelBuilders["com.microsoft.Sampling"]    = sampling_model_builder
SpecialInputGenerators["com.microsoft.Sampling"] = sampling_input_generator
