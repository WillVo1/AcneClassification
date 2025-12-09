# ATTRIBUTION


### Human-Generated
The following components were either contributors and my personal implementation:

- **Core Machine Learning Pipeline** (`acne_classifier/` modules)
  - `model_loader.py` - Model loading and initialization logic
  - `prediction.py` - Acne severity prediction implementation
    - https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py -> Help me implement Insightface
  - `ingredient_recommendations.py` - OpenAI integration for recommendations
  - `product_search.py` - RAG-based product search system (half and half between my work and some support from AI-generated content)
  - `config.py` - Configuration management
  - `__init__.py` - Package initialization

- **Utility Scripts**
  - `Install_Model.py` - Model download and setup script -> received from https://huggingface.co/imfarzanansari/skintelligent-acne?library=transformers
  - `start.sh` - Application startup script
  - `requirements.txt` - Python dependencies
  - `README.md` - Project documentation
  - `SETUP.md` - Installation instructions



### AI-Generated Components

  **Web Frontend** (`web/index.html`)
  - Complete HTML structure
  - Modern CSS styling with gradients and animations
  - JavaScript functionality including:
    - Drag-and-drop file upload
    - Image preview
    - AJAX API communication
    - Dynamic result display
    - Error handling UI

  **Backend Application Logic** (`web/app.py`)
  - Flask application structure
  - API endpoints and routing
  - Error handling and logging
  - Production deployment configuration



### External Data Sources

### Primary Dataset
- **Source**: [Skincare Products and Their Ingredients](https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients/data)
- **Author**: eward96 on Kaggle
- **Description**: Comprehensive dataset of skincare products with ingredient information
- **Usage**: Product recommendation and search functionality
- **License**: [Kaggle Dataset License](https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients/data)
- **File Location**: `data/skincare_products.csv`

## Machine Learning Models

### Primary Classification Model
- **Model Name**: Skintelligent Acne Classifier
- **Source**: [Hugging Face - imfarzanansari/skintelligent-acne](https://huggingface.co/imfarzanansari/skintelligent-acne)
- **Author**: imfarzanansari
- **Model Type**: Vision Transformer (ViT) for Image Classification
- **Purpose**: Acne severity classification (clear, mild, moderate, severe)
- **License**: Check model repository for specific license terms

### Face Detection Model
- **Library**: InsightFace
- **Model**: buffalo_l
- **Purpose**: Automatic face detection and cropping
- **Implementation**: CPU-based execution provider

## Third-Party Libraries and Dependencies

### Core ML and AI Libraries
- **PyTorch** - Deep learning framework for model inference
- **Transformers** (Hugging Face) - Model loading and processing
  - `ViTImageProcessor` - Image preprocessing
  - `ViTForImageClassification` - Classification model
  - `ViTConfig` - Model configuration
- **SafeTensors** - Secure tensor loading
- **InsightFace** - Face detection and analysis
- **OpenAI API** - GPT-based ingredient recommendations
- **scikit-learn** - vectorization and cosine similarity
- **pandas** - Data manipulation and CSV handling
- **numpy** - Numerical computations
- **opencv-python** (cv2) - Image processing
- **Pillow** (PIL) - Image handling

### Web Framework and Utilities
- **Flask** - Web application framework
- **Werkzeug** - WSGI utilities (file upload handling)

### OpenAI API
- **Service**: OpenAI ChatGPT API (GPT-3.5-turbo)
- **Usage**: Based on predicted acne severity level
