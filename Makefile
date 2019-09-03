# simple makefile to simplify repetetive build env management tasks under posix

PYTHON ?= python
PYTESTS ?= py.test

test:
	pytest test_custom_log_regression.py

pep: flake pydocstyle

flake:
	flake8

pydocstyle:
	pydocstyle
