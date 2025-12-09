# Skintelligent Acne Classification System

A simple acne severity classification system that analyzes facial images and provides skincare product recommendations.

## Credits

- **Dataset**: [Skincare Products and Their Ingredients](https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients/data) by eward96 on Kaggle
- **Model**: [Skintelligent Acne Classifier](https://huggingface.co/imfarzanansari/skintelligent-acne) by imfarzanansari on Hugging Face

## How It Works

1. **Face Detection**: Uses InsightFace to automatically detect and crop faces from uploaded images
2. **Acne Classification**: Uses the Skintelligent Vision Transformer (ViT) model from Hugging Face to classify acne severity (clear, mild, moderate, severe)
3. **AI Recommendations**: Generates personalized ingredient recommendations using OpenAI's GPT
4. **Product Search**: RAG-based search system to find relevant skincare products from a database of 1000+ products

## Installation

```bash
pip install -r requirements.txt
```

Set your OpenAI API key (optional):
```bash
export OPENAI_API_KEY='your_key_here'
```

## Usage

### Command Line
```bash
python main.py --image path/to/image.jpg
```

### Web Interface
```bash
cd web
python app.py
```

Then open http://localhost:5000 in your browser.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- OpenCV
- OpenAI API key (optional, for recommendations)

See `requirements.txt` for complete list.

## Severity Levels

- **Clear**: No visible acne
- **Mild**: Few comedones and papules
- **Moderate**: Multiple papules and pustules
- **Severe**: Numerous inflammatory lesions

