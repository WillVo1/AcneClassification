### Core ML Fundamentals

- **Points:** 3
- **Description:** Modular code design with reusable functions and classes rather than monolithic scripts
- **Evidence:**
  - model_loader.py L13–L70
  - prediction.py L58–L84
  - product_search.py L13–L76

---

## Computer Vision

- **Points:** 7
- **Description:** Used or fine-tuned vision transformer architecture (ViT, Swin, DINO, etc.)
- **Evidence:**
  - model_loader.py L3, L31-L37
  - It requires the vision transformers from transformer directly to utilize the pretrain model

- **Points:** 5
- **Description:** Used or fine-tuned vision convolutional neural network architecture
- **Evidence:**
  - model_loader.py L5, L49–L60
  - Please look at this -> https://github.com/deepinsight/insightface/wiki/Tutorial:-Deploy-Face-Recognition-Model-via-TVM
  - It states it is CNN base

---

## Natural Language Processing

- **Points:** 10
- **Description:** Built retrieval-augmented generation (RAG) system with document retrieval (e.g., from a static dataset/database, or from dynamic web search/scraping) and generation components
- **Evidence:**
  - LLM generation: ingredient_recommendations.py L21–34 -> (used to find most valuable ingredient)
  - TF-IDF + similarity retrieval: product_search.py L45–90 -> (sent to RAG to find most valuable products given ingredient from LLM)

- **Points:** 5
- **Description:** Used sentence embeddings for semantic similarity or retrieval
- **Evidence:**
  - product_search.py L47–63, L66–90

- **Points:** 5
- **Description:** Made API calls to state-of-the-art model (GPT-4, Claude, Gemini) with meaningful integration into your system
- **Evidence:**
  - ingredient_recommendations.py L21–34

---

## Advanced System Integration

- **Points:** 7
- **Description:** Built multi-stage ML pipeline connecting outputs of one model to another
- **Evidence:**
  - Vision stage: prediction.py L8–L55
  - LLM ingredient stage: ingredient_recommendations.py L21–L34
  - RAG retrieval stage: product_search.py L27–90

- **Points:** 10
- **Description:** Deployed model as functional web application with user interface
- **Evidence:**
  - app.py: L50-L120
  - index.html: Everything

- **Points:** 10
- **Description:** Implemented production-grade deployment (evidence of at least two considerations such as rate limiting, caching, monitoring, error handling, logging)
- **Evidence:**
  - 

---

## Solo Project Credit

- **Points:** 10
- **Description:** Completed project individually without a partner

**TOTAL: 62**
