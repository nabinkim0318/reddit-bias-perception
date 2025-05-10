# Reddit Bias Perception

A modular pipeline to collect, filter, and analyze Reddit posts for bias in AI-generated images.

The project supports data collection via Reddit API, text preprocessing, few-shot classification (LLM), keyword filtering, sentiment analysis, and topic modeling.

---

## Project Structure

```
reddit_bias_perception/
├── Makefile                        # CLI tasks (install, format, test, etc.)
├── README.md                       # Project documentation
├── .env.example                    # Example environment variables
├── main.py                         # Analysis pipeline (sentiment, topic modeling)
├── pipeline_main.py                # Data collection & preprocessing pipeline
│
├── config/                         # All config files and constants
│   ├── config.py                   # File paths, model configs, constants
│   ├── vectorizer_config.py        # Stopwords and vectorizer settings for topic modeling
│   ├── fewshot_prompt_template.j2  # Prompt template for LLM few-shot classification
│   ├── ai_keywords.json            # Keywords related to AI topics
│   ├── bias_keywords.json          # Keywords grouped by bias type
│   ├── emotion_mapping.json        # Mapping of emotion → sentiment (GoEmotions)
│   └── subreddit_groups.json       # Subreddit groups (e.g., expert vs casual)
│
├── reddit_crawler/                # Reddit API crawling modules
│   ├── reddit_client.py           # Authenticated Reddit API wrapper
│   ├── subreddit_fetcher.py       # Fetch posts & comments from subreddits
│   └── utils.py                   # Helper functions for crawling
│
├── processing/                    # Preprocessing and filtering logic
│   ├── clean_text.py              # Text cleaning (lowercasing, stopwords, etc.)
│   ├── keyword_filter.py          # Filters based on bias and AI keywords
│   ├── llm_few_shot.py            # Few-shot LLM classifier logic
│   └── utils_technical_filter.py  # Optional: removes technical or off-topic content
│
├── analysis/                      # Analysis modules
│   ├── sentiment_analysis.py      # Dual sentiment analysis (GoEmotions + VADER)
│   ├── bertopic_model.py          # Topic modeling using BERTopic
│   ├── topic_mapper.py            # Optional: map topic IDs to names
│   └── emotion/                   # (Optional) Emotion-focused analysis
│       ├── clustering.py
│       ├── manova.py
│       └── mean_plot.py
│
├── visualization/                # Visualization helpers (matplotlib, seaborn)
│   └── plots.py
│
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_data_pipeline_overview.ipynb
│   ├── 02_llm_sample_check.ipynb
│   └── 03_analysis_summary.ipynb
│
├── tests/                        # Unit and integration tests
│   ├── assets/                   # Test fixtures and mock templates
│   │   └── test_prompt_template.j2
│   ├── test_llm_few_shot.py
│   ├── test_llm_few_shot_integration.py
│   ├── test_sentiment_analysis.py
│   ├── test_bertopic_model.py
│   └── ...
│
├── utils/                        # Generic utility functions
│   └── text_utils.py
│
├── data/                         # Local data (ignored by Git)
│   ├── raw/                      # Raw JSON dumps from Reddit
│   ├── processed/                # Cleaned & filtered datasets
│   └── results/                  # Output from sentiment & topic analysis
│       ├── sentiment_labeled_output.csv
│       └── ...
│
├── pyproject.toml                # Poetry configuration
└── poetry.lock                   # Poetry lockfile
'''

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
