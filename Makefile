.DEFAULT_GOAL := help
.PHONY: test system-deps dev-deps deps style lint install help docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks recipes

test_all:	## run tests and don't stop on an error.
	nose2 --with-coverage --coverage TTS tests
	./run_bash_tests.sh

test:	## run tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests

test_vocoder:	## run vocoder tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.vocoder_tests

test_tts:	## run tts tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.tts_tests

test_tts2:	## run tts tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.tts_tests2

test_xtts:
	nose2 -F -v -B --with-coverage --coverage TTS tests.xtts_tests

test_aux:	## run aux tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.aux_tests
	./run_bash_tests.sh

test_zoo:	## run zoo tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.zoo_tests

inference_tests: ## run inference tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.inference_tests

data_tests: ## run data tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.data_tests

test_text: ## run text tests.
	nose2 -F -v -B --with-coverage --coverage TTS tests.text_tests

test_failed:  ## only run tests failed the last time.
	nose2 -F -v -B --with-coverage --coverage TTS tests

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}
	black ${target_dirs} --check
	isort ${target_dirs} --check-only

system-deps:	## install linux system deps
	sudo apt-get install -y libsndfile1-dev

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt

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
