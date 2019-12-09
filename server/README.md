## TTS example web-server

You'll need a model package (Zip file, includes TTS Python wheel, model files, server configuration, and optional nginx/uwsgi configs). Publicly available models are listed [here](https://github.com/mozilla/TTS/wiki/Released-Models).

Instructions below are based on a Ubuntu 18.04 machine, but it should be simple to adapt the package names to other distros if needed. Python 3.6 is recommended, as some of the dependencies' versions predate Python 3.7 and will force building from source, which requires extra dependencies and is not guaranteed to work.

Development server:

1. apt-get install -y espeak libsndfile1 python3-venv
2. python3 -m venv /tmp/venv
3. source /tmp/venv/bin/activate
4. pip install -U pip setuptools wheel
5. # Download model package
6. unzip model.zip
7. pip install -U ./TTS*.whl
8. python -m TTS.server.server

You can now browse to http://localhost:5002

Running with nginx/uwsgi:

1. apt-get install -y uwsgi uwsgi-plugin-python3 nginx espeak libsndfile1 python3-venv
2. python3 -m venv /tmp/venv
3. source /tmp/venv/bin/activate
4. pip install -U pip setuptools wheel
5. # Download model package
6. unzip model.zip
7. pip install -U ./TTS*.whl
8. cp tts_site_nginx /etc/nginx/sites-enabled/default
9. service nginx restart
10. uwsgi --ini uwsgi.ini

You can now browse to http://localhost:80 (edit the port in /etc/nginx/sites-enabled/tts_site_nginx).
Configure number of workers (number of requests that will be processed in parallel) in uwsgi.ini, `processes` setting.
