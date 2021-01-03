setup:
	poetry install

test:
	poetry run pytest tests/test_models.py

ci-test:
	poetry run pytest tests/test_models.py