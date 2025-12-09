import os
from pathlib import Path

# File paths
PROJECT_DIR = Path(__file__).parent.parent
MODEL_CONFIG_PATH = PROJECT_DIR / "pretrain_model/config.json"
PREPROCESSOR_CONFIG_PATH = PROJECT_DIR / "pretrain_model/preprocessor_config.json"
MODEL_WEIGHTS_PATH = PROJECT_DIR / "pretrain_model/model.safetensors"
SKINCARE_DATA_PATH = PROJECT_DIR / "data/skincare_products.csv"

# Model configuration
IMAGE_SIZE = 224
FACE_DETECTION_SIZE = (640, 640)

# Severity mappings
SEVERITY_MAP = {
    'level -1': 'clear_skin',
    'level 0': 'very_mild', 
    'level 1': 'mild',
    'level 2': 'moderate',
    'level 3': 'severe'
}

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Product search configuration
PRODUCT_TYPE_MAPPING = {
    'cleanser': 'Cleanser',
    'moisturizer': 'Moisturiser', 
    'exfoliator': 'Exfoliator'
}

# Search parameters
TOP_K_PRODUCTS = 3
MIN_RELEVANCE_THRESHOLD = 0.1
EXACT_MATCH_BONUS = 0.3
