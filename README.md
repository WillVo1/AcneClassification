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