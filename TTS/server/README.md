<!-- ## TTS example web-server

You'll need a model package (Zip file, includes TTS Python wheel, model files, server configuration, and optional nginx/uwsgi configs). Publicly available models are listed [here](https://github.com/mozilla/TTS/wiki/Released-Models#simple-packaging---self-contained-package-that-runs-an-http-api-for-a-pre-trained-tts-model).

Instructions below are based on a Ubuntu 18.04 machine, but it should be simple to adapt the package names to other distros if needed. Python 3.6 is recommended, as some of the dependencies' versions predate Python 3.7 and will force building from source, which requires extra dependencies and is not guaranteed to work. -->

# :frog: TTS demo server
Before you use the server, make sure you [install](https://github.com/coqui-ai/TTS/tree/dev#install-tts)) :frog: TTS properly. Then, you can follow the steps below.

**Note:** If you install :frog:TTS using ```pip```, you can also use the ```tts-server``` end point on the terminal.

Examples runs:

List officially released models.
```python TTS/server/server.py  --list_models ```

Run the server with the official models.
```python TTS/server/server.py  --model_name tts_models/en/ljspeech/tacotron2-DCA --vocoder_name vocoder_models/en/ljspeech/multiband-melgan```

Run the server with the official models on a GPU.
```CUDA_VISIBLE_DEVICES="0" python TTS/server/server.py  --model_name tts_models/en/ljspeech/tacotron2-DCA --vocoder_name vocoder_models/en/ljspeech/multiband-melgan --use_cuda True```

Run the server with a custom models.
```python TTS/server/server.py  --tts_checkpoint /path/to/tts/model.pth.tar --tts_config /path/to/tts/config.json --vocoder_checkpoint /path/to/vocoder/model.pth.tar --vocoder_config /path/to/vocoder/config.json```
