# Skintelligent Acne Classification System

A acne severity classification system that analyzes facial images and provides skincare product recommendations.

## Credits

- **Dataset**: [Skincare Products and Their Ingredients](https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients/data) by eward96 on Kaggle
- **Model**: [Skintelligent Acne Classifier](https://huggingface.co/imfarzanansari/skintelligent-acne) by imfarzanansari on Hugging Face

## How It Works

1. **Face Detection**: Uses InsightFace to automatically detect and crop faces from uploaded images
2. **Acne Classification**: Uses the Skintelligent Vision Transformer (ViT) model from Hugging Face to classify acne severity (clear, mild, moderate, severe)
3. **AI Recommendations**: Generates personalized ingredient recommendations using OpenAI's GPT
4. **Product Search**: RAG-based search system to find relevant skincare products from a database of 1000+ products

## Installation

## Production Setup

To run the application in production mode:

### 1. Setup Environment Variables
Create your `.env.prod` file with the required environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
PORT=9000
HOST=0.0.0.0
```

### 2. Download the Model
Download the pre-trained model from Hugging Face by running:
```bash
## May take awhile
python install_Model.py
```

### 3. Launch Production Server
Run the startup script to launch the production application:
```bash
./start.sh
```

This script does:
- Create a virtual environment if it doesn't exist
- Install all required dependencies
- Load production environment variables
- Start the server on the configured host and port

## Usage

### Run Web Interface
```bash
cd web
python app.py
```

Then open http://localhost:[Port] in your browser.

### Production Web Interface
After running `./start.sh`, the application will be available at http://127.0.0.1/[port] (or your configured host/port).

## Requirements

See `requirements.txt` for complete list.

## Severity Levels

- **Clear**: No visible acne
- **Mild**: Few comedones and papules
- **Moderate**: Multiple papules and pustules
- **Severe**: Numerous inflammatory lesions
