.PHONY: build-gzip

define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

## The versions of pycaption depended by pycaption and aeneas have no overlapping.
## That will fail setup.py so pip install on requirements.txt is needed.
install:
	if [ ! -e ".venv" ]; then virtualenv -p python3 .venv; fi
	.venv/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .venv/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .venv/bin/pip install; \
	.venv/bin/pip install -e . --ignore-installed
	cp ./bin/subaligner_1pass /usr/local/bin/subaligner_1pass
	cp ./bin/subaligner_2pass /usr/local/bin/subaligner_2pass

uninstall:
	rm -rf /usr/local/bin/subaligner_1pass
	rm -rf /usr/local/bin/subaligner_2pass

build-gzip:
	mkdir -p dist
	tar -czf dist/subligner-cli.tar.gz main bin requirements.txt setup.py tox.ini README.rst HISTORY.rst 

build-rpm:
	mkdir -p BUILD RPMS SRPMS SOURCES BUILDROOT
	tar -czf SOURCES/subligner-cli.tar.gz main requirements.txt

test: ## run tests quickly with the default Python
	if [ ! -e ".venv" ]; then virtualenv -p python3 .venv; fi
	.venv/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .venv/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .venv/bin/pip install
	PYTHONPATH=. .venv/bin/python -m unittest discover
	-.venv/bin/pycodestyle subaligner tests --ignore=E501

test-all: ## run tests on every Python version with tox
	if [ ! -e ".venv" ]; then virtualenv -p python3 .venv; fi
	.venv/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .venv/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .venv/bin/pip install
	.venv/bin/tox

pydoc: clean-doc ## generate pydoc HTML documentation based on docstrings
	python -m pydoc -w subaligner; mv subaligner.html docs/index.html
	python -m pydoc -w subaligner.embedder; mv subaligner.embedder.html docs
	python -m pydoc -w subaligner.media_helper; mv subaligner.media_helper.html docs
	python -m pydoc -w subaligner.network; mv subaligner.network.html docs
	python -m pydoc -w subaligner.predictor; mv subaligner.predictor.html docs
	python -m pydoc -w subaligner.singleton; mv subaligner.singleton.html docs
	python -m pydoc -w subaligner.trainer; mv subaligner.trainer.html docs
	python -m pydoc -w subaligner.utils; mv subaligner.utils.html docs
	python -m pydoc -w subaligner.subtitle; mv subaligner.subtitle.html docs
	python -m pydoc -w subaligner.logger; mv subaligner.logger.html docs
	python -m pydoc -w subaligner.exception; mv subaligner.exception.html docs
	python -m pydoc -w subaligner.models; mv subaligner.models.html docs
	python -m pydoc -w subaligner.models.training; mv subaligner.models.training.html docs
	python -m pydoc -w subaligner.models.training.model; mv subaligner.models.training.model.html docs
	python -m pydoc -w subaligner.models.training.weights; mv subaligner.models.training.weights.html docs
	python -m pydoc -w subaligner._version; mv subaligner._version.html docs

	$(BROWSER) docs/index.html

coverage: ## check code coverage quickly with the default Python
	if [ ! -e ".venv" ]; then virtualenv -p python3 .venv; fi
	.venv/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .venv/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .venv/bin/pip install
	.venv/bin/coverage run --source subaligner -m unittest discover
	.venv/bin/coverage report
	.venv/bin/coverage html
	$(BROWSER) htmlcov/index.html

clean: clean-build clean-pyc clean-test clean-rpm clean-doc ## remove all build, test, coverage and Python artifacts

clean-gzip:
	rm -rf dist

clean-doc: ## remove documents
	rm -rf docs/*.html 

clean-build: clean-rpm ## remove build artifacts
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -fr {} +
	find . -name '*.pyo' -exec rm -fr {} +
	find . -name '*~' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -rf .venv/
	rm -rf .tox/
	rm -fr .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache

clean-rpm:
	rm -rf BUILD RPMS SRPMS SOURCES BUILDROOT