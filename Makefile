.DEFAULT_GOAL = help

#: run linter
lint:
	poetry lock --check
	poetry run autoflake --check .
	poetry run isort --check-only .
	poetry run black --check .
	poetry run mypy  --show-error-codes .
	poetry run pylint epigos

#: format all source files
format:
	poetry run autoflake --in-place .
	poetry run isort --atomic .
	poetry run black .

#: run tests
test:
	poetry run pytest -vvv --cov --cov-report term-missing

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist

hooks:
	poetry run pre-commit install --install-hooks

#: list available make targets
help:
	@grep -B1 -E "^[a-zA-Z0-9_-]+\:([^\=]|$$)" Makefile \
		| grep -v -- -- \
		| sed 'N;s/\n/###/' \
		| sed -n 's/^#: \(.*\)###\(.*\):.*/make \2###    \1/p' \
		| column -t  -s '###' \
		| sort

.PHONY: lint format test clean hooks help
