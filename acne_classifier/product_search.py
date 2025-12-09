import logging
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
        self.logger = logging.getLogger(__name__)
        self.load_data()
    
    def load_data(self):
        try:
            if not SKINCARE_DATA_PATH.exists():
                self.logger.error(f"Skincare data file not found: {SKINCARE_DATA_PATH}")
                self.df = pd.DataFrame()
                return
                
            self.df = pd.read_csv(SKINCARE_DATA_PATH)
            self.df = self.df.dropna(subset=['ingredients', 'product_type'])
            self.logger.info(f"Loaded {len(self.df)} skincare products")
        except Exception as e:
            self.logger.error(f"Error loading skincare data: {e}")
            self.df = pd.DataFrame()
    
    def rag_search(self, target_ingredients, product_type, top_k=TOP_K_PRODUCTS):
        try:
            if self.df.empty:
                self.logger.warning("Product database is empty")
                return []
            
            self.logger.info(f"Searching for {product_type} with ingredients: {target_ingredients}")
            
            # Map product type
            mapped_type = PRODUCT_TYPE_MAPPING.get(product_type.lower())
            if not mapped_type:
                self.logger.error(f"Unknown product type: {product_type}")
                return []
            
            # Filter by product type
            filtered_df = self.df[
                self.df['product_type'].str.contains(mapped_type, case=False, na=False)
            ]
            
            if filtered_df.empty:
                self.logger.warning(f"No products found for type: {mapped_type}")
                return []
            
            self.logger.info(f"Found {len(filtered_df)} products of type {mapped_type}")
            
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
                self.logger.error(f"TF-IDF computation failed: {e}")
                return []
                
        except Exception as e:
            self.logger.error(f"Product search failed: {e}")
            return []
        
        # Score products
        try:
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
            
            sorted_products = sorted(scored_products, key=lambda x: x['combined_score'], reverse=True)[:top_k]
            self.logger.info(f"Found {len(sorted_products)} matching products")
            return sorted_products
            
        except Exception as e:
            self.logger.error(f"Product scoring failed: {e}")
            return []
    
    def search_all_categories(self, ingredient_recommendations):
        try:
            results = {}
            self.logger.info(f"Searching across {len(ingredient_recommendations)} categories")
            
            for category, ingredients in ingredient_recommendations.items():
                try:
                    if ingredients:
                        products = self.rag_search(ingredients, category)
                        results[category] = products
                    else:
                        self.logger.warning(f"No ingredients provided for {category}")
                        results[category] = []
                except Exception as e:
                    self.logger.error(f"Search failed for category {category}: {e}")
                    results[category] = []
            
            return results
        except Exception as e:
            self.logger.error(f"Multi-category search failed: {e}")
            return {}
    
    def format_search_results(self, search_results):
        try:
            output = []
            
            for category, products in search_results.items():
                try:
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
                except Exception as e:
                    self.logger.error(f"Error formatting {category} results: {e}")
                    output.append(f"Error formatting {category} results")
            
            return '\n'.join(output)
        except Exception as e:
            self.logger.error(f"Result formatting failed: {e}")
            return "Error formatting search results"


def rag_search(target_ingredients, product_type, top_k=TOP_K_PRODUCTS):
    searcher = ProductSearcher()
    return searcher.rag_search(target_ingredients, product_type, top_k)
