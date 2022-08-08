ARG BASE=nvcr.io/nvidia/pytorch:22.03-py3
FROM ${BASE}

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv python3-wheel espeak espeak-ng libsndfile1-dev && rm -rf /var/lib/apt/lists>

RUN pip install llvmlite==0.38.1 --ignore-installed
RUN pip install -U pip setuptools wheel

WORKDIR /root
COPY requirements-recod.txt /root
COPY requirements.dev.txt /root
COPY requirements.notebooks.txt /root
RUN ["/bin/bash", "-c", "pip install -r <(cat requirements.txt)"]
COPY . /root