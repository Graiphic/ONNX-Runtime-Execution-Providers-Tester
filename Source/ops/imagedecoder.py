# ops/imagdecoder.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def imagdecoder_model_builder(op_type, cfg=None):
    # Définition de l'entrée
    input_tensor = onnx.helper.make_tensor_value_info("encoded_stream", onnx.TensorProto.UINT8, [None])
    # Définition de la sortie
    output_tensor = onnx.helper.make_tensor_value_info("image", onnx.TensorProto.UINT8, None)

    # Création du nœud ImageDecoder
    node = onnx.helper.make_node(
        "ImageDecoder",
        inputs=["encoded_stream"],
        outputs=["image"],
        pixel_format="RGB"
    )

    # Construction du graphe
    graph = onnx.helper.make_graph(
        nodes=[node],
        name="imagdecoder_graph",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    # Création du modèle
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def imagdecoder_input_generator(session):
    # Génération d'une image aléatoire et encodage en JPEG
    from PIL import Image
    import io

    # Création d'une image RGB aléatoire
    image = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8), 'RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    encoded_image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return {"encoded_stream": encoded_image}

SpecialModelBuilders["ImageDecoder"] = imagdecoder_model_builder
SpecialInputGenerators["ImageDecoder"] = imagdecoder_input_generator
