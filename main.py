import argparse
import sys
import os
from pathlib import Path

# Add the acne_classifier module to the path
sys.path.append(str(Path(__file__).parent / 'acne_classifier'))

from acne_classifier.model_loader import load_models
from acne_classifier.prediction import AcnePredictor
from acne_classifier.ingredient_recommendations import IngredientRecommender
from acne_classifier.product_search import ProductSearcher


def main():
    parser = argparse.ArgumentParser(
        description="Acne Severity Classification and Skincare Recommendation System"
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to the input image file'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    try:
        # Load all models
        print("=" * 50)
        print("ACNE CLASSIFICATION SYSTEM")
        print("=" * 50)
        print("Loading models...")
        
        model, processor, face_app, model_config_dict = load_models()
        
        # Initialize components
        predictor = AcnePredictor(model, processor, face_app, model_config_dict)
        recommender = IngredientRecommender()
        searcher = ProductSearcher()
        
        print(f"Processing image: {args.image}")
        print("-" * 50)
        
        # Step 1: Predict acne severity
        prediction_result = predictor.predict(args.image)
        
        if 'error' in prediction_result:
            print(f"Prediction Error: {prediction_result['error']}")
            sys.exit(1)
        
        # Display prediction results
        print("ACNE SEVERITY PREDICTION:")
        print(f"  Raw Label: {prediction_result['predicted_label']}")
        print(f"  Severity: {prediction_result['severity']}")
        print(f"  Confidence: {prediction_result['confidence']:.4f}")
        
        if args.verbose:
            print(f"  Predicted Class ID: {prediction_result['predicted_class_id']}")
            print(f"  Probabilities: {prediction_result['probabilities']}")
        
        # Step 2: Get ingredient recommendations
        print("\n" + "-" * 50)
        print("INGREDIENT RECOMMENDATIONS:")
        
        recommendations_text = recommender.get_recommendations(prediction_result['predicted_label'])
        print(recommendations_text)
        
        if recommendations_text.startswith("Error"):
            print("Cannot proceed with product search due to recommendation error.")
            sys.exit(1)
        
        # Step 3: Parse recommendations and search products
        parsed_recommendations = recommender.parse_recommendations(recommendations_text)
        
        if args.verbose:
            print(f"\nParsed recommendations: {parsed_recommendations}")
        
        # Step 4: Search for products
        print("\n" + "-" * 50)
        print("PRODUCT SEARCH RESULTS:")
        
        search_results = searcher.search_all_categories(parsed_recommendations)
        formatted_results = searcher.format_search_results(search_results)
        print(formatted_results)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
