import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import (
    SKINCARE_DATA_PATH,
    PRODUCT_TYPE_MAPPING, 
    TOP_K_PRODUCTS, 
    MIN_RELEVANCE_THRESHOLD, 
    EXACT_MATCH_BONUS
)


class ProductSearcher:
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        try:
            self.df = pd.read_csv(SKINCARE_DATA_PATH)
            self.df = self.df.dropna(subset=['ingredients', 'product_type'])
            print(f"Loaded {len(self.df)} skincare products")
        except Exception as e:
            print(f"Error loading skincare data: {e}")
            self.df = pd.DataFrame()
    
    def enhanced_rag_search(self, target_ingredients, product_type, top_k=TOP_K_PRODUCTS):
        if self.df.empty:
            return []
        
        # Map product type
        mapped_type = PRODUCT_TYPE_MAPPING.get(product_type.lower())
        if not mapped_type:
            return []
        
        # Filter by product type
        filtered_df = self.df[
            self.df['product_type'].str.contains(mapped_type, case=False, na=False)
        ]
        
        if filtered_df.empty:
            return []
        
        # Prepare target ingredients query
        target_query = ' '.join([ing.strip().lower() for ing in target_ingredients])
        
        # Create TF-IDF vectors for ingredients
        ingredients_list = [str(ing).lower() for ing in filtered_df['ingredients'].tolist()]
        all_texts = ingredients_list + [target_query]
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between query and products
            query_vector = tfidf_matrix[-1]
            product_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(query_vector, product_vectors).flatten()
            
        except Exception as e:
            print(f"Error in TF-IDF computation: {e}")
            return []
        
        # Score products
        scored_products = []
        for idx, (_, product) in enumerate(filtered_df.iterrows()):
            # Ingredient matching score
            ingredient_text = str(product['ingredients']).lower()
            exact_matches = sum(
                1 for ing in target_ingredients 
                if ing.strip().lower() in ingredient_text
            )
            
            # Combined score: TF-IDF similarity + exact matches
            combined_score = similarities[idx] + (exact_matches * EXACT_MATCH_BONUS)
            
            if combined_score > MIN_RELEVANCE_THRESHOLD:
                scored_products.append({
                    'product_name': product['product_name'],
                    'price': product.get('price', 'N/A'),
                    'similarity_score': round(similarities[idx], 3),
                    'exact_matches': exact_matches,
                    'combined_score': round(combined_score, 3),
                    'url': product.get('product_url', 'N/A'),
                    'ingredients': product['ingredients']
                })
        
        return sorted(scored_products, key=lambda x: x['combined_score'], reverse=True)[:top_k]
    
    def search_all_categories(self, ingredient_recommendations):
        results = {}
        
        for category, ingredients in ingredient_recommendations.items():
            if ingredients:
                products = self.enhanced_rag_search(ingredients, category)
                results[category] = products
            else:
                results[category] = []
        
        return results
    
    def format_search_results(self, search_results):
        output = []
        
        for category, products in search_results.items():
            category_title = category.upper()
            output.append(f"\n=== {category_title} PRODUCTS ===")
            
            if products:
                for i, product in enumerate(products, 1):
                    output.append(f"{i}. {product['product_name']}")
                    output.append(
                        f"   Price: {product['price']} | "
                        f"Matches: {product['exact_matches']} | "
                        f"Score: {product['combined_score']}"
                    )
                    if product['url'] != 'N/A':
                        output.append(f"   URL: {product['url']}")
            else:
                output.append(f"No matching {category} products found.")
        
        return '\n'.join(output)


# Global searcher instance
_product_searcher = None

def get_product_searcher():
    global _product_searcher
    if _product_searcher is None:
        _product_searcher = ProductSearcher()
    return _product_searcher

def enhanced_rag_search(target_ingredients, product_type, top_k=TOP_K_PRODUCTS):
    searcher = get_product_searcher()
    return searcher.enhanced_rag_search(target_ingredients, product_type, top_k)
