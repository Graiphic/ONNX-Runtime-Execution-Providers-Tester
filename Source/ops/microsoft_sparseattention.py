# ops/microsoft_sparseattention.py

import numpy as np
import onnx
import onnx.helper
import onnxruntime as rt
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def sparseattention_model_builder(op_type, cfg=None):
    float_type = onnx.TensorProto.FLOAT
    int_type   = onnx.TensorProto.INT32

    B, S, H, HS, L, BS = 1, 4, 4, 8, 2, 2
    max_seq = 8
    d = (H + 2 * H) * HS

    query = onnx.helper.make_tensor_value_info("query", float_type, [B, S, d])
    block_row_indices = onnx.helper.make_tensor_value_info("block_row_indices", int_type, [L, (max_seq // BS) + 1])
    block_col_indices = onnx.helper.make_tensor_value_info("block_col_indices", int_type, [L, 9])
    total_seq_len = onnx.helper.make_tensor_value_info("total_sequence_length", int_type, [])
    key_total_seq_lens = onnx.helper.make_tensor_value_info("key_total_sequence_lengths", int_type, [B])

    past_key_in  = onnx.helper.make_tensor_value_info("past_key_in",  float_type, [B, H, max_seq, HS])
    past_key_out = onnx.helper.make_tensor_value_info("past_key_out", float_type, [B, H, max_seq, HS])
    past_value_in  = onnx.helper.make_tensor_value_info("past_value_in",  float_type, [B, H, max_seq, HS])
    past_value_out = onnx.helper.make_tensor_value_info("past_value_out", float_type, [B, H, max_seq, HS])
    output = onnx.helper.make_tensor_value_info("output", float_type, [B, S, H * HS])

    node = onnx.helper.make_node(
        "SparseAttention",
        inputs=[
            "query", "", "", "past_key_in", "past_value_in",
            "block_row_indices", "block_col_indices",
            "total_sequence_length", "key_total_sequence_lengths"
        ],
        outputs=["output", "past_key_out", "past_value_out"],
        domain="com.microsoft",
        kv_num_heads=H,
        num_heads=H,
        sparse_block_size=BS,
        do_rotary=0,
        rotary_interleaved=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "SparseAttentionGraph",
        [query, past_key_in, past_value_in, block_row_indices, block_col_indices, total_seq_len, key_total_seq_lens],
        [output, past_key_out, past_value_out]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model


def sparseattention_input_generator(session):
    def iobinding_callback(sess):
        io_binding = sess.io_binding()

        B, S, H, HS, L, BS = 1, 4, 4, 8, 2, 2
        max_seq = 8
        d = (H + 2 * H) * HS

        query_data = np.random.rand(B, S, d).astype(np.float32)
        block_row_indices_data = np.array([[0, 1, 3, 5, 8], [0, 1, 3, 6, 9]], dtype=np.int32)
        block_col_indices_data = np.array([[0, 0, 1, 1, 2, 1, 2, 3, -1], [0, 0, 1, 0, 1, 2, 0, 2, 3]], dtype=np.int32)
        total_sequence_length_data = np.array(max_seq, dtype=np.int32)
        key_total_sequence_lengths_data = np.array([max_seq], dtype=np.int32)

        shared_pk = np.zeros((B, H, max_seq, HS), dtype=np.float32)
        shared_pv = np.zeros((B, H, max_seq, HS), dtype=np.float32)
        ortvalue_query = rt.OrtValue.ortvalue_from_numpy(query_data, "cpu", 0)
        ortvalue_pk = rt.OrtValue.ortvalue_from_numpy(shared_pk, "cpu", 0)
        ortvalue_pv = rt.OrtValue.ortvalue_from_numpy(shared_pv, "cpu", 0)

        def bind_input(name, array, ortvalue):
            io_binding.bind_input(name=name, device_type="cpu", device_id=0,
                                  element_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[array.dtype],
                                  shape=array.shape, buffer_ptr=ortvalue.data_ptr())

        def bind_scalar_input(name, scalar_array):
            io_binding.bind_input(name=name, device_type="cpu", device_id=0,
                                  element_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[scalar_array.dtype],
                                  shape=(), buffer_ptr=rt.OrtValue.ortvalue_from_numpy(scalar_array, "cpu", 0).data_ptr())

        bind_input("query", query_data, ortvalue_query)
        bind_input("past_key_in", shared_pk, ortvalue_pk)
        bind_input("past_value_in", shared_pv, ortvalue_pv)
        bind_input("block_row_indices", block_row_indices_data,
                   rt.OrtValue.ortvalue_from_numpy(block_row_indices_data, "cpu", 0))
        bind_input("block_col_indices", block_col_indices_data,
                   rt.OrtValue.ortvalue_from_numpy(block_col_indices_data, "cpu", 0))
        bind_scalar_input("total_sequence_length", total_sequence_length_data)
        bind_input("key_total_sequence_lengths", key_total_sequence_lengths_data,
                   rt.OrtValue.ortvalue_from_numpy(key_total_sequence_lengths_data, "cpu", 0))

        output_shapes = {out.name: tuple(out.shape) for out in sess.get_outputs()}
        io_binding.bind_output("past_key_out", "cpu", 0,
                               element_type=onnx.TensorProto.FLOAT,
                               shape=output_shapes["past_key_out"],
                               buffer_ptr=ortvalue_pk.data_ptr())
        io_binding.bind_output("past_value_out", "cpu", 0,
                               element_type=onnx.TensorProto.FLOAT,
                               shape=output_shapes["past_value_out"],
                               buffer_ptr=ortvalue_pv.data_ptr())
        out_shape = output_shapes["output"]
        out_buf = np.zeros(out_shape, dtype=np.float32)
        ortvalue_out = rt.OrtValue.ortvalue_from_numpy(out_buf, "cpu", 0)
        io_binding.bind_output("output", "cpu", 0,
                               element_type=onnx.TensorProto.FLOAT,
                               shape=out_shape,
                               buffer_ptr=ortvalue_out.data_ptr())

        sess.run_with_iobinding(io_binding)
        return {
            "output": ortvalue_out.numpy(),
            "past_key_out": shared_pk,
            "past_value_out": shared_pv
        }

    return {"__iobinding__": iobinding_callback}


SpecialModelBuilders["com.microsoft.SparseAttention"] = sparseattention_model_builder
SpecialInputGenerators["com.microsoft.SparseAttention"] = sparseattention_input_generator
