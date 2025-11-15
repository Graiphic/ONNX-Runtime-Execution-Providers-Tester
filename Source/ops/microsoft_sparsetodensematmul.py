# ops/microsoft_sparsetodensematmul.py
import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto
import onnxruntime
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def sparsetodensematmul_model_builder(op_type, cfg=None):
    float_type = TensorProto.FLOAT

    # Déclarations des formes
    a_shape = [3, 4]
    b_shape = [4, 2]

    # A est sparse 3×4, B est dense 4×2, Y résultat dense
    A = onnx.helper.make_sparse_tensor_value_info("A", float_type, a_shape)
    B = onnx.helper.make_tensor_value_info("B", float_type, b_shape)
    Y = onnx.helper.make_tensor_value_info("Y", float_type, None)

    node = onnx.helper.make_node(
        "SparseToDenseMatMul",
        inputs=["A", "B"],
        outputs=["Y"],
        domain="com.microsoft",
        transA=0,
        transB=0,
        alpha=1.0
    )

    graph = onnx.helper.make_graph(
        [node],
        "SparseToDenseMatMulGraph",
        [A, B],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model


def sparsetodensematmul_input_generator(session):
    """
    Génère un feed dict adapté pour SparseToDenseMatMul, en créant un OrtValue
    sparse pour "A" (COO) et un numpy array pour "B".
    """
    # 1) Définir valeurs et indices non-nuls de A
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # Indices COO en 2-D : shape (2, nnz)
    # correspond aux positions (0,1), (1,2), (2,3)
    indices = np.array([
        [0, 1, 2],   # lignes
        [1, 2, 3]    # colonnes
    ], dtype=np.int64)
    shape = (3, 4)

    # 2) Créer un objet SparseTensor en Python (COO) sur CPU
    cpu_device = onnxruntime.OrtDevice.make("cpu", 0)
    A_sparse = onnxruntime.SparseTensor.sparse_coo_from_numpy(
        shape,      # (3,4)
        values,     # np.array([1.0,2.0,3.0], dtype=float32)
        indices,    # np.array([[0,1,2],[1,2,3]], dtype=int64)
        cpu_device  # exécution sur CPU
    )
    # 3) Convertir en OrtValue
    ortvalue_A = onnxruntime.OrtValue.ort_value_from_sparse_tensor(A_sparse)

    # 4) Générer B dense
    B = np.random.rand(4, 2).astype(np.float32)

    return {
        "A": ortvalue_A,  # OrtValue sparse en entrée
        "B": B            # numpy array dense
    }


SpecialModelBuilders["com.microsoft.SparseToDenseMatMul"] = sparsetodensematmul_model_builder
SpecialInputGenerators["com.microsoft.SparseToDenseMatMul"] = sparsetodensematmul_input_generator
