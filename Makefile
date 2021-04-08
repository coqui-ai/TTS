.DEFAULT_GOAL := help
.PHONY: test deps style lint install help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks

deps:	## install 🐸 requirements.
	pip install -r requirements.txt

test:	## run tests.
	nosetests --with-cov -cov  --cover-erase --cover-package TTS tests

style:	## update code style.
	black ${target_dirs}
	isort ${target_dirs}

lint:	## run pylint linter.
	pylint ${target_dirs}

install:	## install 🐸 TTS for development.
	pip install -e .
