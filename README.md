# Reddit Bias Perception

A modular pipeline to collect, filter, and analyze Reddit posts for bias in AI-generated images.

The project supports data collection via Reddit API, text preprocessing, few-shot classification (LLM), keyword filtering, sentiment analysis, and topic modeling.

---

## Project Structure

```
reddit_bias_perception/
├── config/            # Configuration files and constants
├── reddit_crawler/    # Reddit API data collection
├── processing/         # Text cleaning, keyword filtering, LLM classification
├── analysis/           # Sentiment analysis, topic modeling
├── notebooks/          # Exploratory notebooks
├── tests/              # Unit & integration tests
├── data/               # Input/output data files (raw, processed, results)
├── pipeline_main.py    # End-to-end data collection pipeline
├── main.py             # Main analysis pipeline
└── Makefile            # CLI commands (install, test, format, etc.)
```

---

## Pipelines

### 1. Data Collection & Preprocessing (`pipeline_main.py`)

```bash
poetry run python pipeline_main.py
```

This script:

- Crawls posts & comments using Reddit API
- Cleans the text (removes noise, stopwords, etc.)
- Runs few-shot classification via LLM (Gemma 2B)
- Applies keyword filtering for bias + AI relevance

---

### 2. Analysis Pipeline (`main.py`)

```bash
poetry run python main.py
```

This script:

- Performs sentiment analysis (GoEmotions + VADER)
- Runs BERTopic for topic modeling
- Saves labeled results and topic distributions

---

## Development Commands

| Command       | Description                   |
|---------------|-------------------------------|
| `make install`| Install dependencies with Poetry |
| `make format` | Run black and isort            |
| `make check`  | Check formatting and config    |
| `make test`   | Run all tests via pytest       |

---

## Configuration

Environment variables and paths are defined in:

- `config/config.py` → all file paths and constants
- `.env` → API keys and credentials

### Example `.env`
```
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=script:reddit_bias:v1.0 (by u/yourname)
HF_TOKEN=hf_your_token
```

---

## Outputs

| File                     | Description                            |
|--------------------------|----------------------------------------|
| `reddit_raw.json`        | Raw Reddit posts/comments              |
| `reddit_bias_data_clean.csv` | Cleaned text data                  |
| `classified_bias.csv`    | LLM-labeled bias posts                 |
| `filtered_ai_bias.csv`   | Keyword-filtered AI-relevant posts     |
| `sentiment_labeled.csv`  | Sentiment analysis output              |
| `bertopic_post_topics.csv` | BERTopic topic assignments          |

---

## Tests

Test files are under `tests/` and can be run with:

```bash
make test
```

Includes:

- Unit tests for each module
- Integration tests for LLM filtering

---

## License

MIT License. Feel free to fork and extend.
