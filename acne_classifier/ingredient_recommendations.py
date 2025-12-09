import logging
import openai
import base64
from .config import OPENAI_API_KEY


def get_ingredient_recommendations(severity):
    # Get skincare ingredient recommendations from OpenAI based on acne severity
    logger = logging.getLogger(__name__)
    
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured")
        return "Error: OpenAI API key not configured"
    
    logger.info(f"Getting recommendations for severity: {severity}")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Request ingredient recommendations using GPT-3.5
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user", 
                    "content": f"For {severity} acne, pick only 1 ingredient that best suits severity level: 1) Cleanser 2) Moisturizer 3) Exfoliator? Provide only ingredient names separated by commas for each category. Each line should be [Category]: [Ingredient] (This is a plan so please make a treatment for each possible ingredient unique per severity, you are allowed to list non-acne treatment ingredients)."
                }
            ]
        )
        
        logger.info("Successfully received OpenAI recommendations")
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error getting recommendations: {str(e)}"


def parse_ingredient_line(line):
    # Parse a single line of ingredient recommendations into a list
    line = line.strip()
    if not line:
        return []
    
    # Handle format like "Cleanser: ingredient1, ingredient2" or just "ingredient1, ingredient2"
    if ':' in line:
        parts = line.split(':', 1)
        if len(parts) > 1:
            ingredients = parts[1].strip()
        else:
            ingredients = parts[0].strip()
    
    # Split comma-separated ingredients and remove whitespace
    if ingredients:
        return [ing.strip() for ing in ingredients.split(',') if ing.strip()]
    return []


class IngredientRecommender:
    # Class wrapper for ingredient recommendation functionality
    def __init__(self):
        self.client = None
        self.logger = logging.getLogger(__name__)
        if OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def get_recommendations(self, predicted_label):
        # Get ingredient recommendations for a given severity level
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return "Error: OpenAI API key not configured"
        
        return get_ingredient_recommendations(predicted_label)
    
    def parse_recommendations(self, recommendations_text):
        # Parse OpenAI response into structured dictionary by product category
        try:
            parsed = {
                'cleanser': [],
                'moisturizer': [],
                'exfoliator': []
            }
            
            # Process each line and categorize by product type
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
        except Exception as e:
            self.logger.error(f"Failed to parse recommendations: {str(e)}")
            return str(e)
    
    def generate_daily_plan(self, severity, ingredient_recommendations, product_results):
        # RAG: Generate personalized daily skincare plan using retrieved products as context -> AI generated Method
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return "Error: OpenAI API key not configured"
        
        self.logger.info(f"Generating daily plan for severity: {severity}")
        
        try:
            # Build context from retrieved products
            context = self._build_context(severity, ingredient_recommendations, product_results)
            
            # Generate personalized plan using LLM with retrieved context (RAG)
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a dermatology expert creating personalized skincare routines. Use the provided product recommendations to create a detailed, easy-to-follow daily plan."
                    },
                    {
                        "role": "user",
                        "content": f"""Based on the following information, create a detailed daily skincare routine:

                    Acne Severity: {severity}

                    Recommended Ingredients:
                    {ingredient_recommendations}

                    Available Products:
                    {context}

                    Please create a comprehensive daily skincare routine with:
                    1. Morning Routine (step-by-step)
                    2. Evening Routine (step-by-step)
                    3. Additional Tips specific to {severity} acne

                    Be specific about which products to use when, and include the product names from the recommendations above."""
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            plan = response.choices[0].message.content
            self.logger.info("Successfully generated daily skincare plan")
            return plan
            
        except Exception as e:
            self.logger.error(f"Plan generation error: {str(e)}")
            return f"Error generating plan: {str(e)}"
    
    def _build_context(self, severity, ingredients, product_results):
        # Build context string from retrieved products for RAG
        context_parts = []
        
        # Add severity and ingredient context
        context_parts.append(f"Acne Severity Level: {severity}")
        context_parts.append(f"\nRecommended Ingredients:\n{ingredients}")
        context_parts.append("\nRetrieved Products:\n")
        
        # Add product information from each category
        for category, products in product_results.items():
            if products:
                context_parts.append(f"\n{category.upper()}:")
                for i, product in enumerate(products, 1):
                    context_parts.append(
                        f"  {i}. {product['product_name']} "
                        f"(Exact matches: {product['exact_matches']}, "
                        f"Relevance: {product['combined_score']})"
                    )
            else:
                context_parts.append(f"\n{category.upper()}: No specific products found")
        
        return '\n'.join(context_parts)
