ARG BASE=nvcr.io/nvidia/pytorch:22.03-py3
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-wheel espeak libsndfile1-dev && rm -rf /var/lib/apt/lists/*

# Create and activate virtual env
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip setuptools wheel

WORKDIR /root
COPY requirements.txt /root
COPY requirements.dev.txt /root
COPY requirements.notebooks.txt /root
RUN ["/bin/bash", "-c", "pip install -r <(cat requirements.txt requirements.dev.txt requirements.notebooks.txt)"]
COPY . /root
RUN make install
ENTRYPOINT ["tts"]
CMD ["--help"]
