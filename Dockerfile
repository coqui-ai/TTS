FROM nvcr.io/nvidia/pytorch:22.03-py3
RUN apt-get update && apt-get install -y --no-install-recommends espeak && rm -rf /var/lib/apt/lists/*
WORKDIR /root
COPY requirements.txt /root
COPY requirements.dev.txt /root
COPY requirements.notebooks.txt /root
RUN pip install -r <(cat requirements.txt requirements.dev.txt requirements.notebooks.txt)
COPY . /root
RUN make install
ENTRYPOINT ["tts"]
CMD ["--help"]
