lint:
	pylint src tests/unit

cov:
	pytest --cov=src --cov-report term-missing --cov-fail-under 80 tests/unit
