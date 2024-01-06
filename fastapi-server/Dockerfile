FROM python:3.10

RUN mkdir -p /code/test
WORKDIR /code
COPY demo.py test/default_speaker.json /code/
COPY test/ /code/test/
RUN apt-get -y update && apt-get -y install ffmpeg
RUN pip install gradio requests
CMD ["python", "demo.py"]
