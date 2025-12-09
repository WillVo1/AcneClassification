import logging
import torch
import cv2
import numpy as np
from PIL import Image
from .config import SEVERITY_MAP, IMAGE_SIZE


def predict_image(image_path, model, processor, face_app, model_config_dict):
    # Main function to predict acne severity from an image using face detection and classification
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting prediction for image: {image_path}")
        
        # Load and convert image to RGB format
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Convert PIL image to numpy array for face detection
        img_rgb = np.array(image)
        
        # Detect faces using InsightFace
        faces = face_app.get(img_rgb)
        
        if len(faces) == 0:
            logger.warning("No face detected in the image")
            return {'error': 'No face detected in the image'}
        
        logger.info(f"Detected {len(faces)} face(s)")
        
        # Extract bounding box coordinates from the first detected face
        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        
        # Validate bounding box coordinates
        if x1 >= x2 or y1 >= y2:
            logger.error("Invalid face bounding box detected")
            return {'error': 'Invalid face detection'}
        
        # Crop face region and resize to model input size
        face_crop = img_rgb[y1:y2, x1:x2]
        face_crop_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))
        face_image = Image.fromarray(face_crop_resized)
        
        # Preprocess face image for model input
        inputs = processor(images=face_image, return_tensors="pt")
        
        # Run model inference without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process model outputs to get predictions and probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = logits.argmax().item()
        predicted_label = model_config_dict['id2label'][str(predicted_class_id)]
        confidence = probabilities[0][predicted_class_id].item()
        
        logger.info(f"Prediction completed: {predicted_label} (confidence: {confidence:.4f})")
        
        return {
            'raw_logits': logits.squeeze().numpy(),
            'probabilities': probabilities.squeeze().numpy(),
            'predicted_class_id': predicted_class_id,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'severity': SEVERITY_MAP[str(predicted_label)]
        }
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {str(e)}")
        return {'error': 'Image file not found'}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f'Prediction failed: {str(e)}'}


class AcnePredictor:
    # Class wrapper for acne prediction with loaded models
    def __init__(self, model, processor, face_app, model_config_dict):
        self.model = model
        self.processor = processor
        self.face_app = face_app
        self.model_config_dict = model_config_dict
        self.logger = logging.getLogger(__name__)
    
    def predict(self, image_path):
        # Predict acne severity for a single image
        try:
            return predict_image(
                image_path, 
                self.model, 
                self.processor, 
                self.face_app, 
                self.model_config_dict
            )
        except Exception as e:
            self.logger.error(f"Predictor error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}
