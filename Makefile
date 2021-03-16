help:
	@echo 'make install: install package, requirements and pre-commit hook'

install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .
	pre-commit install

lint:
	flake8 .

update:
	pre-commit autoupdate --bleeding-edge

cov:
	pytest --cov=modes

mypy:
	mypy . --ignore-missing-imports

lint:
	tox -e flake8

pylint:
	pylint --rcfile .pylintrc pp/

lintdocs:
	flake8 --select RST

lintdocs2:
	pydocstyle modes

doc8:
	doc8 docs/

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

codestyle:
	pycodestyle --max-line-length=88
