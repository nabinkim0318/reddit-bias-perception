install:
	poetry install

lock:
	poetry lock && poetry install

format:
	poetry run black . && poetry run isort .

check:
	poetry check && poetry run black --check . && poetry run isort --check-only .

test:
	PYTHONPATH=. poetry run pytest tests/ --disable-warnings

pipeline:
	poetry run python pipeline_main.py
