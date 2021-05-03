.DEFAULT_GOAL := help
.PHONY: test system-deps dev-deps deps style lint install help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks

system-deps:	## install linux system deps
	sudo apt-get install -y espeak-ng
	sudo apt-get install -y libsndfile1-dev

dev-deps:  ## install development deps
	pip install -r requirements.dev.txt
	pip install -r requirements.tf.txt

deps:	## install 🐸 requirements.
	pip install -r requirements.txt

test:	## run tests.
	nosetests -x --with-cov -cov  --cover-erase --cover-package TTS tests --nologcapture
	./run_bash_tests.sh

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}

install:	## install 🐸 TTS for development.
	pip install -e .[all]
