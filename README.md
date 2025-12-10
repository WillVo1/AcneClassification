# GRADER

I made some last minute changes, please redownload the Assessment! RAG System enquired changes that are not reflected in the video! I added last step API call at the end to make a daily plan for each user with the products -> I created this to reflect the RAG system due to some misunderstanding of my orignal RAG System and system errors that I have been struggling with since last week.
Thank you!


# Skintelligent Acne Classification System

## Goal
Problem: When I first got into skincare due to puberty (lots of acne), I had 0 idea what Salicylic acid or Benzoyle meant when I was reading the bottles. Oftentimes, these acne treatment ingredients are made according to the severity of the acne and I had to do a signifcant amount of research just to understand who Salicylic acid is for. It is complicated to decide what skin products (cleanser, moisturizer, exfoliators) best fit my acne severity without damaging the skin signifcantly (too much treatment is bad!).

Goal: Due to personal experiences, I wanted to create a website that can detect the severity of your acne and find the best acne ingredients. These ingredients will significantly cut down the time for researching and finding the best suiting products without worrying about over-treatment. My product will output some suggested products given the best ingredient for your specific severity and allow you to build a skincare routine.

## What it Does

The Skintelligent Acne Classification System is an AI-powered skincare solution that analyzes facial images to classify acne severity and provides personalized skincare product recommendations. The system uses InsightFace for automatic face detection and cropping, employs a Vision Transformer (ViT) model from Hugging Face to classify acne severity into four categories (clear, mild, moderate, severe), generates AI-powered ingredient recommendations using OpenAI's GPT, and implements a RAG-based search system to find skincare products from a database of over 1000 products that will produce a day to day schedule.

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
1. Clone the repository and navigate to the project directory
2. Create a `.env.prod` file with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_ENV=production
   PORT=9000
   HOST=0.0.0.0
   ```
3. Download the pre-trained model:
   ```bash
   python Install_Model.py
   ```
4. Launch the application:
   ```bash
   ./start.sh
   ```
5. Open your browser and navigate to `http://127.0.0.1:9000` to access the web interface

## Video Links

- **Demo Video**: https://drive.google.com/file/d/1JRmW_77CtzUK7BCl4riSuuOROkh0mZcv/view?usp=sharing
- **Technical Walkthrough**: https://drive.google.com/file/d/1UMjn0fB9WHOmNnyq6I8k7LtUUQjJRI8P/view?usp=sharing


## Design Reasons

I wanted a simple upload and receive system that can generate products and a schedule that perfectly aligns with most people. So my core principle here is convenience, thus building a website and a simple upload and click button. I was only limited to products that is available in the set and there is only so much variety that the openai model can output, so I intended to focus on single ingredients to maximize the accuracy of products I can get. When there is too many ingredients per category, it can reduce accuracy from my rag system and confuse the generated plan. So after some testing, I decided a single best ingredient was best for each category and built a system that best serve customers