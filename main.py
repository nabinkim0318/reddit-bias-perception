"""Compatibility shim. Prefer the canonical module command.

Canonical offline demo::

    python -m processing.run_pipeline --synthetic \\
        --input tests/fixtures/synthetic/posts.json \\
        --output-dir artifacts/synthetic_demo

This file no longer runs sentiment analysis or BERTopic.
"""

from processing.run_pipeline import main

if __name__ == "__main__":
    main()
