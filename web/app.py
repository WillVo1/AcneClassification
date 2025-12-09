from flask import Flask, request, render_template, jsonify
import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import traceback

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir / 'acne_classifier'))
sys.path.append(str(parent_dir))

from acne_classifier.model_loader import load_models
from acne_classifier.prediction import AcnePredictor
from acne_classifier.ingredient_recommendations import IngredientRecommender
from acne_classifier.product_search import ProductSearcher

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

predictor = None
recommender = None
searcher = None

def initialize_models():
    """Initialize all models on startup"""
    global predictor, recommender, searcher
    try:
        print("Loading models...")
        # Change to parent directory for model loading
        original_cwd = os.getcwd()
        os.chdir(parent_dir)
        
        model, processor, face_app, model_config_dict = load_models()
        
        # Change back to web directory
        os.chdir(original_cwd)
        
        # Initialize components
        predictor = AcnePredictor(model, processor, face_app, model_config_dict)
        recommender = IngredientRecommender()
        searcher = ProductSearcher()
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG images.'}), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Step 1: Predict acne severity
            prediction_result = predictor.predict(temp_path)
            
            if 'error' in prediction_result:
                return jsonify({'error': f"Prediction Error: {prediction_result['error']}"}), 400
            
            # Step 2: Get ingredient recommendations
            recommendations_text = recommender.get_recommendations(prediction_result['predicted_label'])
            
            if recommendations_text.startswith("Error"):
                return jsonify({'error': f"Recommendation Error: {recommendations_text}"}), 400
            
            # Step 3: Parse recommendations and search products
            parsed_recommendations = recommender.parse_recommendations(recommendations_text)
            search_results = searcher.search_all_categories(parsed_recommendations)
            formatted_results = searcher.format_search_results(search_results)
            
            # Format the complete result
            result = {
                'prediction': {
                    'severity': prediction_result['severity'],
                    'confidence': round(prediction_result['confidence'], 4),
                    'raw_label': prediction_result['predicted_label']
                },
                'recommendations': recommendations_text,
                'products': formatted_results
            }
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Acne Classification Web Application...")
    
    if not initialize_models():
        print("Failed to load models. Exiting.")
        sys.exit(1)
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
