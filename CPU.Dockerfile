ARG PYTHON_VERSION="3.9"
FROM python:${PYTHON_VERSION} AS builder

# Create and activate virtual env
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip setuptools wheel

WORKDIR /root
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /root
COPY requirements.dev.txt /root
COPY requirements.notebooks.txt /root
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends gcc g++ make python3-wheel espeak espeak-ng libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install llvmlite --ignore-installed \
    && pip install -r requirements.txt \
    && pip install -r requirements.dev.txt \
    && pip install -r requirements.notebooks.txt
COPY . /root
RUN pip install -e .[all]


##################
# Slim python
##################
FROM python:${PYTHON_VERSION}-slim 
WORKDIR /root
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
COPY . /root
RUN pip install -e .[all]
ENTRYPOINT ["tts"]
CMD ["--help"]