help:
	@echo 'make install:          install package, requirements and pre-commit hook'

install:
	pip install -r requirements_dev.txt
	pip install -e .
	pre-commit install

lint:
	flake8 .

publish:
	rm -rf dist
	python setup.py sdist bdist_wheel
	twine upload dist/*
