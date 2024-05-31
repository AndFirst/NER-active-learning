.PHONY: build app clean data lint format requirements sync-data test test-cov docs help

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = zprp-ner-active-learning
PYTHON_INTERPRETER = python

build: clean venv requirements format lint test

venv:
	virtualenv .venv

app: format
	. .venv/bin/activate; \
	$(PYTHON_INTERPRETER) -m app.app; \
	deactivate

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.so" -delete
	find . -type d -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	if [ -d "docs/_build" ]; then rm -rf docs/build; fi
	if [ -d "htmlcov" ]; then rm -rf htmlcov; fi
	find . -type d -name "*.egg-info" -exec rm -rf {} +

	rm -rf .pytest_cache

lint:
	. .venv/bin/activate; \
	flake8 --max-line-length=130 app/ tests/ --exclude=__init__.py; \
	deactivate

format:
	. .venv/bin/activate; \
	black --line-length 130 --target-version py310 app/ tests/; \
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place app/ tests/ --exclude=__init__.py; \
	deactivate

test:
	. .venv/bin/activate; \
	pytest; \
	deactivate

test-cov: format
	. .venv/bin/activate; \
	pytest --cov-report term-missing:skip-covered --cov=app; \
	deactivate

docs:
	make -C docs html

docs-clean:
	make -C docs clean

requirements:
	. .venv/bin/activate; \
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel; \
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt; \
	deactivate

help:
	@echo "clean        remove all build, test, coverage and Python artifacts"
	@echo "lint         check style with flake8"
	@echo "black        format code with black"
	@echo "test         run tests quickly with the default Python"
	@echo "test-cov     check code coverage quickly with the default Python"
	@echo "docs         generate Sphinx HTML documentation, including API docs"
	@echo "requirements install the Python dependencies"