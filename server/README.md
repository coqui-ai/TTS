## TTS example web-server

You'll need a model package (Zip file, includes TTS Python wheel, model files, server configuration, and optional nginx/uwsgi configs). Publicly available models are listed [here](https://github.com/mozilla/TTS/wiki/Released-Models#simple-packaging---self-contained-package-that-runs-an-http-api-for-a-pre-trained-tts-model).

Instructions below are based on a Ubuntu 18.04 machine, but it should be simple to adapt the package names to other distros if needed. Python 3.6 is recommended, as some of the dependencies' versions predate Python 3.7 and will force building from source, which requires extra dependencies and is not guaranteed to work.

#### Development server:

##### Using server.py
If you have the environment set already for TTS, then you can directly call ```setup.py```.

##### Using .whl
1. apt-get install -y espeak libsndfile1 python3-venv
2. python3 -m venv /tmp/venv
3. source /tmp/venv/bin/activate
4. pip install -U pip setuptools wheel
5. pip install -U https//example.com/url/to/python/package.whl
6. python -m TTS.server.server

You can now open http://localhost:5002 in a browser

#### Running with nginx/uwsgi:

1. apt-get install -y uwsgi uwsgi-plugin-python3 nginx espeak libsndfile1 python3-venv
2. python3 -m venv /tmp/venv
3. source /tmp/venv/bin/activate
4. pip install -U pip setuptools wheel
5. pip install -U https//example.com/url/to/python/package.whl
6. curl -LO https://github.com/reuben/TTS/releases/download/t2-ljspeech-mold/t2-ljspeech-mold-nginx-uwsgi.zip
7. unzip *-nginx-uwsgi.zip
8. cp tts_site_nginx /etc/nginx/sites-enabled/default
9. service nginx restart
10. uwsgi --ini uwsgi.ini

You can now open http://localhost:80 in a browser (edit the port in /etc/nginx/sites-enabled/tts_site_nginx).
Configure number of workers (number of requests that will be processed in parallel) by editing the `uwsgi.ini` file, specifically the `processes` setting.

#### Creating a server package with an embedded model

[setup.py](../setup.py) was extended with two new parameters when running the `bdist_wheel` command:

- `--checkpoint <path to checkpoint file>` - path to model checkpoint file you want to embed in the package
- `--model_config <path to config.json file>` - path to corresponding config.json file for the checkpoint

To create a package, run `python setup.py bdist_wheel --checkpoint /path/to/checkpoint --model_config /path/to/config.json`.

A Python `.whl` file will be created in the `dist/` folder with the checkpoint and config embedded in it.
