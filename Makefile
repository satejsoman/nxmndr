.PHONY: clean clean-build clean-pyc clean-test lint test test-unit test-integration coverage install help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with ruff and mypy
	cd nxmndr && ruff check src/
	cd nxmndr && mypy src/

format: ## auto-format code with ruff
	cd nxmndr && ruff format src/ tst/
	cd nxmndr && ruff check --fix src/ tst/

test: ## run all tests
	cd nxmndr && pytest -q

test-unit: ## run unit tests (no external services or GPU required)
	cd nxmndr && pytest -m unit -q

test-integration: ## run integration tests (spins up ephemeral gRPC servers)
	cd nxmndr && pytest -m integration -q

coverage: ## check code coverage
	cd nxmndr && pytest --cov=src/nxmndr --cov-report=term-missing --cov-report=html
	@echo "Coverage report: nxmndr/htmlcov/index.html"

install: ## install the nxmndr server package in editable mode
	cd nxmndr && pip install -e .[server,dev]

audit: ## run pip-audit to check for known vulnerabilities
	cd nxmndr && pip-audit
