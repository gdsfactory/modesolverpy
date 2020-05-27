help:
	@echo 'make install:          install package, requirements and pre-commit hook'

install:
	pip install -e .
	pip install pre-commit
	pre-commit install

lint:
	flake8 .

publish:
	python setup.py sdist bdist_wheel
	twine upload dist/*
