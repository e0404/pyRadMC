.PHONY: test test-all lint format types bench clean hooks

hooks:
	git config core.hooksPath .githooks

test:
	pytest

test-all:
	pytest -m ""

validation:
	pytest -m validation

bench:
	pytest -m perf --benchmark-only

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

types:
	mypy pyRadMC

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
