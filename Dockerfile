FROM nvidia/cuda:9.0-base-ubuntu16.04 as base

WORKDIR /srv/app

RUN apt-get update && \
	apt-get install -y git software-properties-common wget vim build-essential libsndfile1 && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get update && \
	apt-get install -y python3.6 python3.6-dev python3.6-tk && \
	# Install pip manually
	wget https://bootstrap.pypa.io/get-pip.py && \
	python3.6 get-pip.py && \
	rm get-pip.py && \
	# Used by the server in server/synthesizer.py
	pip install soundfile

ADD . /srv/app

# Setup for development
RUN python3.6 setup.py develop

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

CMD python3.6 server/server.py -c server/conf.json
