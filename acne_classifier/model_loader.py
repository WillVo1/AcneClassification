import json
import logging
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from safetensors.torch import load_file
from insightface.app import FaceAnalysis
from .config import (
    MODEL_CONFIG_PATH, 
    PREPROCESSOR_CONFIG_PATH, 
    MODEL_WEIGHTS_PATH,
    FACE_DETECTION_SIZE
)

class ModelLoader:    
    def __init__(self):
        self.model = None
        self.processor = None
        self.face_app = None
        self.model_config_dict = None
        self.logger = logging.getLogger(__name__)
    
    def load_acne_model(self):
        try:
            self.logger.info("Loading acne classification model...")
            
            # Check if model files exist
            if not MODEL_CONFIG_PATH.exists():
                raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")
            if not PREPROCESSOR_CONFIG_PATH.exists():
                raise FileNotFoundError(f"Preprocessor config not found: {PREPROCESSOR_CONFIG_PATH}")
            if not MODEL_WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
            
            # Load model configuration
            with open(MODEL_CONFIG_PATH, 'r') as f:
                self.model_config_dict = json.load(f)
            
            # Load preprocessor configuration
            with open(PREPROCESSOR_CONFIG_PATH, 'r') as f:
                preprocessor_config = json.load(f)
            
            # Initialize model
            model_config = ViTConfig(**self.model_config_dict)
            self.model = ViTForImageClassification(config=model_config)
            
            # Load weights
            self.model.load_state_dict(load_file(MODEL_WEIGHTS_PATH))
            self.model.eval()
            
            # Initialize processor
            processor_params = {
                k: v for k, v in preprocessor_config.items() 
                if k != 'image_processor_type'
            }
            self.processor = ViTImageProcessor(**processor_params)
            
            self.logger.info("Acne classification model loaded successfully!")
            return self.model, self.processor
            
        except Exception as e:
            self.logger.error(f"Failed to load acne model: {str(e)}")
            raise
    
    def load_face_detection(self):
        try:
            self.logger.info("Loading face detection model...")
            
            self.face_app = FaceAnalysis(
                name="buffalo_l", 
                providers=['CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=FACE_DETECTION_SIZE)
            
            self.logger.info("Face detection model loaded successfully!")
            return self.face_app
            
        except Exception as e:
            self.logger.error(f"Failed to load face detection model: {str(e)}")
            raise
    
    def load_all_models(self):
        try:
            self.load_acne_model()
            self.load_face_detection()
            self.logger.info("All models loaded successfully")
            return self.model, self.processor, self.face_app, self.model_config_dict
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise


def load_models():
    loader = ModelLoader()
    return loader.load_all_models()
