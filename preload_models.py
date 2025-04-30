import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from layers import (
    EfficientChannelAttention,
    FixedSpatialAttention,
    FixedHybridBlock
)

# Set mixed precision policy first
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

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

# Create proper dummy input matching model architecture
dummy_input = [
    np.zeros((1, 224, 224, 3)),  # Full image input
    np.zeros((1, 224, 224, 3))   # Face image input
]
model.predict(dummy_input, verbose=0)

print("All models preloaded successfully!")
