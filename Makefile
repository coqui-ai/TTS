.DEFAULT_GOAL := help
.PHONY: test system-deps dev-deps deps style lint install help docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks recipes

test_all:	## run tests and don't stop on an error.
	nosetests --with-cov -cov  --cover-erase --cover-package TTS tests --nologcapture --with-id
	./run_bash_tests.sh

test:	## run tests.
	nosetests -x --with-cov -cov  --cover-erase --cover-package TTS tests --nologcapture --with-id

test_vocoder:	## run vocoder tests.
	nosetests tests.vocoder_tests -x --with-cov -cov  --cover-erase --cover-package TTS tests.vocoder_tests --nologcapture --with-id

test_tts:	## run tts tests.
	nosetests tests.tts_tests -x --with-cov -cov  --cover-erase --cover-package TTS tests.tts_tests --nologcapture --with-id

test_aux:	## run aux tests.
	nosetests tests.aux_tests -x --with-cov -cov  --cover-erase --cover-package TTS tests.aux_tests --nologcapture --with-id
	./run_bash_tests.sh

test_zoo:	## run zoo tests.
	nosetests tests.zoo_tests -x --with-cov -cov  --cover-erase --cover-package TTS tests.zoo_tests --nologcapture --with-id

test_failed:  ## only run tests failed the last time.
	nosetests -x --with-cov -cov  --cover-erase --cover-package TTS tests --nologcapture --failed

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}

system-deps:	## install linux system deps
	sudo apt-get install -y libsndfile1-dev

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt
	pip install -r requirements.tf.txt

doc-deps:  ## install docs dependencies
	pip install -r docs/requirements.txt

build-docs: ## build the docs
	cd docs && make clean && make build

hub-deps:  ## install deps for torch hub use
	pip install -r requirements.hub.txt

deps:	## install 🐸 requirements.
	pip install -r requirements.txt

install:	## install 🐸 TTS for development.
	pip install -e .[all]

docs:	## build the docs
	$(MAKE) -C docs clean && $(MAKE) -C docs html
