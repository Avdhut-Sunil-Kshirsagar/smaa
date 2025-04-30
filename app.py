import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
import shutil

# Configure environment before any imports
os.environ.update({
    'DEEPFACE_HOME': '/app/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'DEEPFACE_CACHE': '0'
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import layers, models
from deepface import DeepFace
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from layers import (
    EfficientChannelAttention,
    FixedSpatialAttention,
    FixedHybridBlock
)

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images with build-time initialization",
    version="1.0"
)

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_WORKERS = 2

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_exists: bool

class RequestResourceManager:
    """Manages resources for a single request"""
    def __init__(self):
        self.temp_arrays = []
    
    def add_temp_array(self, arr):
        self.temp_arrays.append(arr)
    
    def cleanup(self):
        tf.keras.backend.clear_session()
        for arr in self.temp_arrays: del arr
        self.temp_arrays.clear()

def is_valid_image(file_bytes):
    """Validate image format"""
    try: return imghdr.what(None, h=file_bytes) in ['jpeg', '
