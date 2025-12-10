### Category 1

### Core ML Fundamentals

- **Points:** 3
- **Description:** Modular code design with reusable functions and classes rather than monolithic scripts
- **Evidence:**
  - model_loader.py
  - prediction.py
  - product_search.py
  - ingredient_recommendations.py

---

## Computer Vision

- **Points:** 7
- **Description:** Used or fine-tuned vision transformer architecture (ViT, Swin, DINO, etc.)
- **Evidence:**
  - model_loader.py L3, L31-L37
  - It requires the ViT from transformer directly to utilize the pretrain model

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
  - LLM generation: ingredient_recommendations.py L21–34 -> (used to find most valuable ingredients per category)
  - similarity retrieval: product_search.py L129–L140 -> (sent to RAG to find most valuable products given ingredient from LLM)

- **Points:** 5
- **Description:** Used sentence embeddings for semantic similarity or retrieval
- **Evidence:**
  - product_search.py L69-L84

- **Points:** 5
- **Description:** Made API calls to state-of-the-art model (GPT-4, Claude, Gemini) with meaningful integration into your system
- **Evidence:**
  - ingredient_recommendations.py L25–39

---

## Advanced System Integration

- **Points:** 7
- **Description:** Built multi-stage ML pipeline connecting outputs of one model to another
- **Evidence:**
  - Cropping: prediction.py L24-L26 (Face Analysis from Insightface model -> cropped image)
  - Vision stage: prediction.py L45-L60 (cropped image fed into SkinIntelligence Model)
  - LLM ingredient stage: app.py L120-L122 (SkinIntelligence Model output -> OpenAI Model generates Ingredient List)

- **Points:** 10
- **Description:** Deployed model as functional web application with user interface
- **Evidence:**
  - app.py: L50-L120
  - index.html

- **Points:** 10
- **Description:** Implemented production-grade deployment (evidence of at least two considerations such as rate limiting, caching, monitoring, error handling, logging)
- **Evidence:**
  - Thorough logging and error handling in product_search.py, prediction.py, model_loader.py, and ingredient_recommendations.py
  - After running, should produce web/app.logs for logs

---

## Solo Project Credit

- **Points:** 10
- **Description:** Completed project individually without a partner

**Total: 72 Points :D**
**Total Marked: 10/15**


### Category 2 ###

## Submission and Self-Assessment (3 points each)

- **Points:** 3
- **Description:** Ontime submission by 5 pm on Friday, December 5th Tuesday, December 9th (note that late submissions will be accepted but only for the normal 72 hour late period, and will not qualify for this rubric item).

- **Points:** 3
- **Description:** Self-assessment submitted that follows guidelines for at most 15 selections in Machine Learning with evidence (note that failing to submit a self-assessment may result in a loss of credit for some overlooked rubric items during grading).
- **Evidence** 10/15 rubrics marked
---

## Basic Documentation (2 points each)

- **Points:** 2
- **Description:** SETUP.md exists with clear, step-by-step installation instructions

- **Points:** 2
- **Description:** ATTRIBUTION.md exists with detailed attributions of all sources including AI-generation information

- **Points:** 2
- **Description:** requirements.txt or environment.yml file is included and accurate

## README.md (1 point each)

- **Points:** 3
- **Description** - README.md has What it Does section that describes in one paragraph what your project does
                    README.md has Quick Start section that concisely explains how to run your project
                    README.md has Video Links section with direct links to your demo and technical walkthrough videos

---

## Video Submissions (2 points each)

- **Points:** 4
- **Description:** Demo video is of the correct length and appropriate for non-specialist audience with no code shown
                  Technical walkthrough is of the correct length and clearly explains code structure, ML techniques, and key contributions
- **Evidence** There is no description of "correct length" in final_project.html so I am assuming less than 10 mins.

---

## Project Workshop Days (1 points each)

**Points** 2
  -Attended 1-2 project workshop days
  -Attended 3-4 project workshop days

**Total: 21 Points :D**


### Category 3 ###

## Project Purpose and Motivation (3 points each)

- **Points:** 3
- **Description:** README clearly articulates a single, unified project goal or research question
- **Evidence** Goal in README

- **Points:** 3
- **Description:** Project demo video effectively communicates why the project matters to a non-technical audience in non-technical terms

- **Points:** 3
- **Description:** Project addresses a real-world problem or explores a meaningful research question
- **Evidence** Problem in README

---

## Technical Coherence (3 points each)

- **Points:** 3
- **Description:** Technical walkthrough demonstrates how components work together synergistically (not just isolated experiments)

- **Points:** 3
- **Description:** Project shows clear progression from problem → approach → solution → evaluation
- **Evidence** Described in demo video

- **Points:** 3
- **Description:** Design choices are explicitly justified in videos or documentation
- **Evidence** Describe in demo video and README

- **Points:** 3
- **Description:** None of the major components awarded rubric item credit in the machine learning category are superfluous to the larger goals of the project (no unrelated "point collecting")

- **Points:** 3
- **Description:** Clean codebase with readable code and no extraneous, stale, or unused files

**Total: 24 Points :D**

72 / 70 -> 70
21 / 20 -> 20
24 / 20 -> 20

**Total Score: 110 Points (Pretty Please :D)**

