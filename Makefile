PYTHON ?= python
PYTEST ?= pytest

.PHONY: help install test parse-dump clean

help:
	@echo "Available targets:"
	@echo "  make install     - Install development and test dependencies"
	@echo "  make test        - Run automated tests with pytest"
	@echo "  make parse-dump  - Run Wikipedia dump stream parser"
	@echo "  make clean       - Remove cache and temporary files"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTEST) tests/ -v

parse-dump:
	$(PYTHON) -m bsearch.data.dump_parser --help

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
