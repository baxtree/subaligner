ifdef PYTHON
PYTHON := $(PYTHON)
else
PYTHON := 3.8.2
endif

ifdef PLATFORM
PLATFORM := $(PLATFORM)
else
PLATFORM := linux-x86_64-cp-37-cp37
endif

SUBALIGNER_VERSION := $(SUBALIGNER_VERSION)
TRIGGER_URL := ${TRIGGER_URL}

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

.PHONY: install uninstall build-gzip build-rpm test test-all docker-build pydoc coverage manual dist release clean clean-dist clean-doc clean-manual clean-build clean-pyc clean-test clean-rpm

install:
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	.$(PYTHON)/bin/pip install -e . --ignore-installed
	cp ./bin/subaligner_1pass .$(PYTHON)/bin/subaligner_1pass
	cp ./bin/subaligner_2pass .$(PYTHON)/bin/subaligner_2pass
	cp ./bin/subaligner_batch .$(PYTHON)/bin/subaligner_batch
	cp ./bin/subaligner_convert .$(PYTHON)/bin/subaligner_convert
	cp ./bin/subaligner_train .$(PYTHON)/bin/subaligner_train
	cp ./bin/subaligner_tune .$(PYTHON)/bin/subaligner_tune
	cp ./bin/subaligner .$(PYTHON)/bin/subaligner

uninstall:
	rm -f .$(PYTHON)/bin/subaligner
	rm -f .$(PYTHON)/bin/subaligner_1pass
	rm -f .$(PYTHON)/bin/subaligner_2pass
	rm -f .$(PYTHON)/bin/subaligner_batch
	rm -f .$(PYTHON)/bin/subaligner_convert
	rm -f .$(PYTHON)/bin/subaligner_train
	rm -f .$(PYTHON)/bin/subaligner_tune

install-basic:
	.$(PYTHON)/bin/pip install -e '.' --no-cache-dir

install-translation:
	.$(PYTHON)/bin/pip install -e '.[llm]' --no-cache-dir

install-stretch:
	.$(PYTHON)/bin/pip install -e '.[stretch]' --no-cache-dir

install-dev:
	.$(PYTHON)/bin/pip install -e '.[dev]' --no-cache-dir

install-docs:
	.$(PYTHON)/bin/pip install -e '.[docs]' --no-cache-dir

install-harmony:
	.$(PYTHON)/bin/pip install -e '.[harmony]' --no-cache-dir

build-gzip:
	mkdir -p dist
	tar -czf dist/subligner.tar.gz subaligner bin requirements.txt setup.py README.md LICENCE

build-rpm:
	mkdir -p BUILD RPMS SRPMS SOURCES BUILDROOT
	tar -czf SOURCES/subligner.tar.gz subaligner bin requirements.txt setup.py README.md LICENCE

test:
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install
	PYTHONPATH=. .$(PYTHON)/bin/python -m unittest discover
	-.$(PYTHON)/bin/pycodestyle subaligner tests examples misc bin/subaligner bin/subaligner_1pass bin/subaligner_2pass bin/subaligner_batch bin/subaligner_convert bin/subaligner_train  bin/subaligner_tune setup.py --ignore=E203,E501,W503 --exclude="subaligner/lib"

test-all: ## run tests on every Python version with tox
	.$(PYTHON)/bin/tox

test-int: ## integration test
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install
	.$(PYTHON)/bin/pip install -e . --ignore-installed
	( \
		source .$(PYTHON)/bin/activate; \
		radish -b tests/integration/radish tests/integration/feature; \
	)

pydoc: clean-doc ## generate pydoc HTML documentation based on docstrings
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	.$(PYTHON)/bin/python -m pydoc -w subaligner; mv subaligner.html docs/index.html
	.$(PYTHON)/bin/python -m pydoc -w subaligner.embedder; mv subaligner.embedder.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.exception; mv subaligner.exception.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.hparam_tuner; mv subaligner.hparam_tuner.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.hyperparameters; mv subaligner.hyperparameters.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.logger; mv subaligner.logger.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.media_helper; mv subaligner.media_helper.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.network; mv subaligner.network.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.predictor; mv subaligner.predictor.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.singleton; mv subaligner.singleton.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.subtitle; mv subaligner.subtitle.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.trainer; mv subaligner.trainer.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.translator; mv subaligner.translator.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner.utils; mv subaligner.utils.html docs
	.$(PYTHON)/bin/python -m pydoc -w subaligner._version; mv subaligner._version.html docs
	$(BROWSER) docs/index.html

coverage: ## check code coverage quickly with the default Python
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install
	.$(PYTHON)/bin/coverage run --source subaligner -m unittest discover
	.$(PYTHON)/bin/coverage report
	.$(PYTHON)/bin/coverage html
	$(BROWSER) htmlcov/index.html

manual: clean-manual ## generate manual pages
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements-site.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
    SPHINXAPIDOC=../.$(PYTHON)/bin/sphinx-apidoc SPHINXBUILD=../.$(PYTHON)/bin/sphinx-build make -C ./site html; \
	.$(PYTHON)/bin/python -m sphinx -T -b html -d ./site/build/doctrees -D language=en ./site/source ./site/build/html; \
	$(BROWSER) ./site/build/html/index.html

test-dist:
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	.$(PYTHON)/bin/pip install -e .

dist: clean-dist test-dist
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	.$(PYTHON)/bin/python setup.py sdist bdist_wheel bdist_egg

release:
	.$(PYTHON)/bin/twine upload dist/*

pipenv-install:
	pipenv install
	pipenv shell

profile:
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-stretch.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-llm.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	.$(PYTHON)/bin/python -c "import misc.profiler; misc.profiler.generate_profiles()"
	.$(PYTHON)/bin/kernprof -v -l ./misc/profiler.py

app: clean-wheels
	if [ ! -e ".$(PYTHON)" ]; then ~/.pyenv/versions/$(PYTHON)/bin/python3 -m venv .$(PYTHON); fi
	.$(PYTHON)/bin/pip install --upgrade pip setuptools wheel; \
	cat requirements-dev.txt | xargs -L 1 .$(PYTHON)/bin/pip install; \
	.$(PYTHON)/bin/pip wheel --no-cache-dir --wheel-dir=./wheels -r requirements.txt -r requirements-stretch.txt -r requirements-llm.txt; \
	STRETCH_OFF=True .$(PYTHON)/bin/python setup.py bdist_wheel -d ./wheels; \
	.$(PYTHON)/bin/pex subaligner==$(SUBALIGNER_VERSION) --repo=./wheels --platform $(PLATFORM) --no-pypi --no-build --python-shebang="/usr/bin/env python3" -e subaligner -o subaligner-$(PLATFORM).app; \

docker-build:
	docker build --build-arg RELEASE_VERSION=$(SUBALIGNER_VERSION) -f docker/Dockerfile-Ubuntu20 .
	docker build --build-arg RELEASE_VERSION=$(SUBALIGNER_VERSION) -f docker/Dockerfile-ArchLinux .
	docker build --build-arg RELEASE_VERSION=$(SUBALIGNER_VERSION) -f docker/Dockerfile-CentOS7 .
	docker build --build-arg RELEASE_VERSION=$(SUBALIGNER_VERSION) -f docker/Dockerfile-Debian11 .
	docker build --build-arg RELEASE_VERSION=$(SUBALIGNER_VERSION) -f docker/Dockerfile-Fedora37 .

docker-images:
	SUBALIGNER_VERSION=$(SUBALIGNER_VERSION) docker-compose -f ./docker/docker-compose.yml build

docker-push:
	curl -H "Content-Type: application/json" --data '{"source_type": "Tag", "source_name": "v$(SUBALIGNER_VERSION)"}' -X POST $(TRIGGER_URL)

clean: clean-build clean-pyc clean-test clean-rpm clean-doc clean-manual clean-dist ## remove all build, test, coverage and Python artifacts

clean-dist:
	rm -rf dist

clean-doc: ## remove documents
	rm -rf docs/*.html

clean-manual: ## remove manual pages
	rm -rf site/build
	rm -f site/source/subaligner.rst site/source/modules.rst

clean-build: clean-rpm ## remove build artifacts
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -fr {} +
	find . -name '*.pyo' -exec rm -fr {} +
	find . -name '*~' -exec rm -fr {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -rf .$(PYTHON)/
	rm -rf .tox/
	rm -fr .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache

clean-rpm:
	rm -rf BUILD RPMS SRPMS SOURCES BUILDROOT

clean-docker-images:
	docker rmi -f $(docker images --filter=reference='*/subaligner' -qa)

clean-wheels:
	rm -rf wheels