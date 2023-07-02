FROM ubuntu:22.04

RUN apt -y update
RUN apt -y upgrade
RUN apt -y install --no-install-recommends pip ffmpeg openjdk-19-jre-headless

RUN mkdir /a/
ADD requirements*.txt /a/
WORKDIR /a/
RUN pip install -r requirements.txt -r requirements.dev.txt -r requirements.notebooks.txt
RUN pip install seaborn pydub notebook

RUN apt -y install --no-install-recommends gcc libpython3.10-dev

ADD runtime.sh /a/

WORKDIR /a/TTS/
CMD /a/runtime.sh
