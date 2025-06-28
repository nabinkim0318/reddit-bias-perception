# ======== Project Setup ========
install:
	poetry install
	pre-commit install

setup:
	poetry shell
	pre-commit install

# ======== Development ========
lock:
	poetry lock && poetry install

format:
	poetry run black . && poetry run isort .

check:
	poetry check && poetry run black --check . && poetry run isort --check-only .


# ======== Pipeline ========
pipeline:
	poetry run python pipeline_main.py


# ======== Testing ========
test:
	PYTHONPATH=. poetry run pytest tests/ -m "not slow" --disable-warnings

test-all:
	PYTHONPATH=. poetry run pytest tests/ --disable-warnings
