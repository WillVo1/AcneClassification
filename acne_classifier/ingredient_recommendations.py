import openai
from .config import OPENAI_API_KEY


def get_ingredient_recommendations(predicted_label):
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key not configured"
    
    # Map prediction to severity for more descriptive prompt
    severity_map = {
        'level -1': 'clear_skin',
        'level 0': 'very_mild', 
        'level 1': 'mild',
        'level 2': 'moderate',
        'level 3': 'severe'
    }
    
    severity = severity_map.get(str(predicted_label), 'unknown')
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content": f"For {severity} acne, pick only one ingredient that best suits severity level: 1) Cleanser 2) Moisturizer 3) Exfoliator? Provide only ingredient names separated by commas for each category. Each line should be [Category]: [Ingredient]"
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"


def parse_ingredient_line(line):
    line = line.strip()
    if not line:
        return []
    
    # Handle format like "Cleanser: ingredient1, ingredient2"
    if ':' in line:
        parts = line.split(':', 1)
        if len(parts) > 1:
            ingredients = parts[1].strip()
        else:
            ingredients = parts[0].strip()
    
    # Split ingredients by comma and clean them
    if ingredients:
        return [ing.strip() for ing in ingredients.split(',') if ing.strip()]
    return []


class IngredientRecommender:    
    def __init__(self):
        self.client = None
        if OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def get_recommendations(self, predicted_label):
        if not self.client:
            return "Error: OpenAI API key not configured"
        
        return get_ingredient_recommendations(predicted_label)
    
    def parse_recommendations(self, recommendations_text):
        
        parsed = {
            'cleanser': [],
            'moisturizer': [],
            'exfoliator': []
        }
        
        lines = recommendations_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            if 'cleanser' in line_lower:
                parsed['cleanser'] = parse_ingredient_line(line)
            elif 'moisturizer' in line_lower:
                parsed['moisturizer'] = parse_ingredient_line(line)
            elif 'exfoliator' in line_lower:
                parsed['exfoliator'] = parse_ingredient_line(line)
        
        return parsed
