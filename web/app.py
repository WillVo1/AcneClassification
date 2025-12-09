import logging
import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import traceback
from datetime import datetime

from flask import Flask, request, render_template, jsonify

# Add project paths
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir / 'acne_classifier'))
sys.path.append(str(parent_dir))

from acne_classifier.model_loader import load_models
from acne_classifier.prediction import AcnePredictor
from acne_classifier.ingredient_recommendations import IngredientRecommender
from acne_classifier.product_search import ProductSearcher

# Production Flask app
app = Flask(__name__, template_folder='.')
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    DEBUG=False
)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global models
predictor = None
recommender = None
searcher = None
models_loaded = False

def init_models():
    """Initialize models with error handling"""
    global predictor, recommender, searcher, models_loaded
    
    if models_loaded:
        return True
    
    try:
        logger.info("Loading models...")
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        
        model, processor, face_app, model_config_dict = load_models()
        os.chdir(original_cwd)
        
        predictor = AcnePredictor(model, processor, face_app, model_config_dict)
        recommender = IngredientRecommender()
        searcher = ProductSearcher()
        
        models_loaded = True
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if models_loaded and predictor is not None:
        return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
    return jsonify({'status': 'unhealthy'}), 503

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not models_loaded:
            return jsonify({'error': 'Service not ready'}), 503
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Predict
            prediction_result = predictor.predict(temp_path)
            if 'error' in prediction_result:
                return jsonify({'error': prediction_result['error']}), 400
            
            # Get recommendations
            recommendations = recommender.get_recommendations(prediction_result['severity'])
            if recommendations.startswith("Error"):
                return jsonify({'error': 'Recommendation failed'}), 400
            
            # RAG Search: Get products and daily plan together
            parsed_recommendations = recommender.parse_recommendations(recommendations)
            rag_results = searcher.search_all_categories(
                parsed_recommendations,
                severity=prediction_result['severity'],
                recommendations_text=recommendations
            )
            
            # Format product results
            formatted_results = searcher.format_search_results(rag_results['products'])
            
            result = {
                'prediction': {
                    'severity': prediction_result['severity'],
                    'confidence': round(prediction_result['confidence'], 4)
                },
                'recommendations': recommendations,
                'products': formatted_results,
                'daily_plan': rag_results['daily_plan']
            }
            
            logger.info(f"Prediction successful: {prediction_result['severity']}")
            return jsonify(result)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    logger.info("Starting production app...")
    
    if not init_models():
        logger.error("Failed to initialize models")
        sys.exit(1)
    
    # Production settings
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    app.run(host=host, port=port, debug=False)
