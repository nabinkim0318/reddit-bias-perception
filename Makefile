# ======== Project Setup ========
install:
	poetry install

setup:
	poetry shell

# ======== Development ========
lock:
	poetry lock && poetry install

format:
	poetry run black . && poetry run isort .

check:
	poetry check && poetry run black --check . && poetry run isort --check-only .


# ======== Canonical offline demo ========
# Synthetic fixtures only. No Reddit credentials, private data, network, GPU,
# or real LLM. Output is a governed aggregate + provenance manifest.
demo:
	PYTHONPATH=. poetry run python -m processing.run_pipeline --synthetic \
		--input tests/fixtures/synthetic/posts.json \
		--output-dir artifacts/synthetic_demo

# ======== Testing ========
test:
	PYTHONPATH=. poetry run pytest tests/ -m "not slow and not integration_external" --disable-warnings

test-all:
	PYTHONPATH=. poetry run pytest tests/ --disable-warnings
