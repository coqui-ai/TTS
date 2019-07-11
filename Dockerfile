FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /srv/app

RUN apt-get update && \
	apt-get install -y libsndfile1 espeak && \
	apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy Source later to enable dependency caching
COPY requirements.txt /srv/app/
RUN pip install -r requirements.txt

COPY . /srv/app

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

CMD python3.6 server/server.py -c server/conf.json
