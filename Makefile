.PHONY: build venv app clean lint format requirements test test-cov docs docs-clean help

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = zprp-ner-active-learning
PYTHON_INTERPRETER = python3.11

ACTIVATE = . .venv/bin/activate
DEACTIVATE = deactivate


build: clean venv requirements format lint test

venv:
	$(PYTHON_INTERPRETER) -m venv .venv

app: format
	$(ACTIVATE)
	$(PYTHON_INTERPRETER) -m app.app

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
	$(ACTIVATE)
	-flake8 --max-line-length=130 app/ tests/ --exclude=__init__.py

format:
	$(ACTIVATE)
	black --line-length 130 --target-version py310 app/ tests/
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place app/ tests/ --exclude=__init__.py

test:
	$(ACTIVATE)
	pytest

test-cov: format
	$(ACTIVATE)
	pytest --cov-report term-missing:skip-covered --cov=app

docs:
	make -C docs html

docs-clean:
	make -C docs clean

requirements:
	$(ACTIVATE) && $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(ACTIVATE) && $(PYTHON_INTERPRETER) -m pip install -r requirements.txt

help:
	@echo "build         build the project"
	@echo "venv          create a virtual environment"
	@echo "app           run the app"
	@echo "clean         remove all build, test, coverage and Python artifacts"
	@echo "lint          check style with flake8"
	@echo "format        format code with black and autoflake"
	@echo "requirements  install the Python dependencies"
	@echo "test          run tests quickly with the default Python"
	@echo "test-cov      check code coverage quickly with the default Python"
	@echo "docs          generate Sphinx HTML documentation, including API docs"
	@echo "docs-clean    clean the docs"
	@echo "help          display this help message"