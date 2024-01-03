FROM python:3.10

RUN mkdir /code
WORKDIR /code
COPY demo.py test/default_speaker.json /code/
RUN mkdir /code/test/
COPY test/ /code/test/
RUN pip install gradio requests
CMD ["python", "demo.py"]
