# Acne Classification System

A modular acne severity classification system that combines computer vision, AI recommendations, and product search capabilities.

## Features

- **Acne Severity Prediction**: Uses a Vision Transformer (ViT) model to classify acne severity from facial images
- **Face Detection**: Automatically detects and crops faces from input images
- **AI-Powered Recommendations**: Generates ingredient recommendations using OpenAI's GPT API
- **Product Search**: RAG-based search system to find relevant skincare products
- **Modular Architecture**: Clean separation of concerns with dedicated modules

## Project Structure

```
skintelligent-acne/
├── acne_classifier/           # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration and constants
│   ├── model_loader.py       # Model loading utilities
│   ├── prediction.py         # Acne severity prediction
│   ├── ingredient_recommendations.py  # OpenAI ingredient recommendations
│   └── product_search.py     # RAG-based product search
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── config.json              # Model configuration
├── preprocessor_config.json  # Image preprocessing config
├── model.safetensors        # Pre-trained model weights
├── skincare_products.csv    # Product database
└── README.md               # This file
```

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (optional, but required for ingredient recommendations):
```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

## Usage

### Command Line Interface

Basic usage:
```bash
python main.py --image path/to/your/image.jpg
```

With verbose output:
```bash
python main.py --image path/to/your/image.jpg --verbose
```

### Example Output

```
==================================================
ACNE CLASSIFICATION SYSTEM
==================================================
Loading models...
Loading acne classification model...
Acne classification model loaded successfully!
Loading face detection model...
Face detection model loaded successfully!
Loaded 1000 skincare products
Processing image: test_image.jpg
--------------------------------------------------
ACNE SEVERITY PREDICTION:
  Raw Label: level 1
  Severity: mild
  Confidence: 0.8542

--------------------------------------------------
INGREDIENT RECOMMENDATIONS:
Cleanser: Salicylic Acid
Moisturizer: Niacinamide
Exfoliator: Glycolic Acid

--------------------------------------------------
PRODUCT SEARCH RESULTS:

=== CLEANSER PRODUCTS ===
1. CeraVe Foaming Facial Cleanser
   Price: $12.99 | Matches: 1 | Score: 0.856
   
=== MOISTURIZER PRODUCTS ===
1. The Ordinary Niacinamide 10% + Zinc 1%
   Price: $7.20 | Matches: 1 | Score: 0.923

=== EXFOLIATOR PRODUCTS ===
1. Paula's Choice SKIN PERFECTING 2% BHA
   Price: $32.00 | Matches: 1 | Score: 0.765

==================================================
ANALYSIS COMPLETE
==================================================
```

## Module Documentation

### config.py
Contains all configuration constants, file paths, and system settings.

### model_loader.py
- `ModelLoader`: Class for loading ML models
- `load_models()`: Convenience function to load all required models

### prediction.py
- `AcnePredictor`: Main prediction class
- `predict_image()`: Core prediction function
- `map_prediction_to_severity()`: Maps raw predictions to human-readable severity

### ingredient_recommendations.py
- `IngredientRecommender`: OpenAI-powered recommendation system
- `get_ingredient_recommendations()`: Get AI recommendations for acne severity
- `parse_ingredient_line()`: Parse recommendation text

### product_search.py
- `ProductSearcher`: RAG-based product search system
- `enhanced_rag_search()`: TF-IDF based similarity search
- Supports searching across cleanser, moisturizer, and exfoliator categories

## Severity Levels

The system classifies acne into 5 severity levels:
- **Level -1**: Clear skin (no acne)
- **Level 0**: Very mild acne
- **Level 1**: Mild acne
- **Level 2**: Moderate acne
- **Level 3**: Severe acne

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- OpenCV
- InsightFace (for face detection)
- OpenAI API key (for recommendations)
- See `requirements.txt` for complete list

## Error Handling

The system includes comprehensive error handling for:
- Missing or invalid image files
- Face detection failures
- Missing OpenAI API keys
- Model loading errors
- Product database issues

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required for ingredient recommendations)

## Limitations

- Requires clear facial images for accurate acne detection
- Face must be visible and well-lit
- Product recommendations depend on the quality of the CSV database
- OpenAI API usage incurs costs

## Troubleshooting

1. **"No face detected"**: Ensure the image contains a clear, visible face
2. **"OpenAI API key not configured"**: Set the OPENAI_API_KEY environment variable
3. **Model loading errors**: Ensure all model files are present and accessible
4. **Import errors**: Verify all dependencies are installed via `pip install -r requirements.txt`

## Contributing

This modular architecture makes it easy to extend the system:
- Add new prediction models by extending `model_loader.py`
- Implement different recommendation systems in `ingredient_recommendations.py`
- Enhance product search algorithms in `product_search.py`
- Add new interfaces by creating alternative entry points to `main.py`
