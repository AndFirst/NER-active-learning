.PHONY: build app clean data lint format requirements sync-data test test-cov docs help

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = zprp-ner-active-learning
PYTHON_INTERPRETER = python

build: clean install format lint test

app: format
	$(PYTHON_INTERPRETER) -m app.app

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.so" -delete
	find . -type d -name "*.pyd" -delete
	find docs/_build -exec rm -r {} +
	rm -rf .pytest_cache

lint:
	flake8 src/ app/ tests/

format:
	black --line-length 79 --target-version py310 app/ src/ tests/
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place app/ src/ tests/ --exclude=__init__.py

test:
	pytest

test-cov: format
	pytest --cov-report term-missing --cov=src

docs:
	make -C docs html

docs-clean:
	make -C docs clean

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

help:
	@echo "clean        remove all build, test, coverage and Python artifacts"
	@echo "lint         check style with flake8"
	@echo "black        format code with black"
	@echo "test         run tests quickly with the default Python"
	@echo "test-cov     check code coverage quickly with the default Python"
	@echo "docs         generate Sphinx HTML documentation, including API docs"
	@echo "requirements install the Python dependencies"