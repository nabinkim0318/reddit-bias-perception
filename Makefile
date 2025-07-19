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


# ======== Pipeline ========
pipeline:
	poetry run python pipeline_main.py


# ======== Testing ========
test:
	PYTHONPATH=. python -m pytest tests/ -m "not slow" --disable-warnings

test-all:
	PYTHONPATH=. python -m pytest tests/ --disable-warnings
