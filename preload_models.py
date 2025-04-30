import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from layers import (
    EfficientChannelAttention,
    FixedSpatialAttention,
    FixedHybridBlock
)

# Initialize DeepFace models
print("Preloading DeepFace models...")
dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
DeepFace.extract_faces(
    img_path=dummy_img,
    detector_backend='opencv',
    enforce_detection=False,
    align=False
)

# Load and warm up main model
print("Preloading main classification model...")
custom_objects = {
    'EfficientChannelAttention': EfficientChannelAttention,
    'FixedSpatialAttention': FixedSpatialAttention,
    'FixedHybridBlock': FixedHybridBlock
}

model = tf.keras.models.load_model(
    os.environ['MODEL_PATH'],
    custom_objects=custom_objects
)
dummy_input = [np.zeros((1, 224, 224, 3)), np.zeros((1, 224, 224, 3))]
model.predict(dummy_input, verbose=0)

print("All models preloaded successfully!")
