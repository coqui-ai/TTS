(synthesizing_speech)=
# Synthesizing Speech

First, you need to install TTS. We recommend using PyPi. You need to call the command below:

```bash
$ pip install TTS
```

After the installation, 2 terminal commands are available.

1. TTS Command Line Interface (CLI). - `tts`
2. Local Demo Server. - `tts-server`
3. In 🐍Python. - `from TTS.api import TTS`

## On the Commandline - `tts`
![cli.gif](https://github.com/coqui-ai/TTS/raw/main/images/tts_cli.gif)

After the installation, 🐸TTS provides a CLI interface for synthesizing speech using pre-trained models. You can either use your own model or the release models under 🐸TTS.

Listing released 🐸TTS models.

```bash
tts --list_models
```

Run a TTS model, from the release models list, with its default vocoder. (Simply copy and paste the full model names from the list as arguments for the command below.)

```bash
tts --text "Text for TTS" \
    --model_name "<type>/<language>/<dataset>/<model_name>" \
    --out_path folder/to/save/output.wav
```

Run a tts and a vocoder model from the released model list. Note that not every vocoder is compatible with every TTS model.

```bash
tts --text "Text for TTS" \
    --model_name "<type>/<language>/<dataset>/<model_name>" \
    --vocoder_name "<type>/<language>/<dataset>/<model_name>" \
    --out_path folder/to/save/output.wav
```

Run your own TTS model (Using Griffin-Lim Vocoder)

```bash
tts --text "Text for TTS" \
    --model_path path/to/model.pth \
    --config_path path/to/config.json \
    --out_path folder/to/save/output.wav
```

Run your own TTS and Vocoder models

```bash
tts --text "Text for TTS" \
    --config_path path/to/config.json \
    --model_path path/to/model.pth \
    --out_path folder/to/save/output.wav \
    --vocoder_path path/to/vocoder.pth \
    --vocoder_config_path path/to/vocoder_config.json
```

Run a multi-speaker TTS model from the released models list.

```bash
tts --model_name "<type>/<language>/<dataset>/<model_name>"  --list_speaker_idxs  # list the possible speaker IDs.
tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx "<speaker_id>"
```

**Note:** You can use ```./TTS/bin/synthesize.py``` if you prefer running ```tts``` from the TTS project folder.

## On the Demo Server - `tts-server`

 <!-- <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/demo_server.gif" height="56"/> -->
![server.gif](https://github.com/coqui-ai/TTS/raw/main/images/demo_server.gif)

You can boot up a demo 🐸TTS server to run an inference with your models. Note that the server is not optimized for performance
but gives you an easy way to interact with the models.

The demo server provides pretty much the same interface as the CLI command.

```bash
tts-server -h # see the help
tts-server --list_models  # list the available models.
```

Run a TTS model, from the release models list, with its default vocoder.
If the model you choose is a multi-speaker TTS model, you can select different speakers on the Web interface and synthesize
speech.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>"
```

Run a TTS and a vocoder model from the released model list. Note that not every vocoder is compatible with every TTS model.

```bash
tts-server --model_name "<type>/<language>/<dataset>/<model_name>" \
           --vocoder_name "<type>/<language>/<dataset>/<model_name>"
```

## Python API

You can run a multi-speaker and multi-lingual model in Python as

```python
from TTS.api import TTS

# List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[0]
# Init TTS
tts = TTS(model_name)
# Run TTS
# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")
```

Here is an example for a single speaker model.

```python
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)
```