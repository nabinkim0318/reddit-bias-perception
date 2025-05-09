# Reddit Bias Sentiment

This project performs dual sentiment analysis on Reddit posts related to AI bias using:

- 🤗 [GoEmotions RoBERTa](https://huggingface.co/j-hartmann/emotion-english-roberta-base) (converted to ONNX for faster inference)
- 🔍 [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for rule-based analysis

## Features

- Batch inference with ONNX-accelerated transformer model
- VADER sentiment scoring
- Visualization of emotional and polarity distributions
- Clean modular structure for extension

## Installation

```bash
poetry install
Make sure you're using a compatible Python version (e.g., >=3.9 <3.10).

Usage
Run sentiment analysis:

bash
Copy
Edit
poetry run python analysis/sentiment_analysis.py
Environment Configuration
Create a .env file and define:

env
Copy
Edit
SENTIMENT_INPUT=data/processed/ai_bias_final.csv
SENTIMENT_OUTPUT=data/results/sentiment_labeled.csv
Project Structure
bash
Copy
Edit
analysis/
│
├── sentiment_analysis.py  # Main sentiment script using ONNX + VADER
├── config/                # Environment variables, constants
├── utils/                 # Reusable functions (preprocessing, IO, etc.)
├── data/
│   ├── processed/         # Input CSVs
│   └── results/           # Output CSVs