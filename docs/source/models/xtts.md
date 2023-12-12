# ⓍTTS
ⓍTTS is a super cool Text-to-Speech model that lets you clone voices in different languages by using just a quick 3-second audio clip. Built on the 🐢Tortoise,
ⓍTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy.
There is no need for an excessive amount of training data that spans countless hours.

This is the same model that powers [Coqui Studio](https://coqui.ai/), and [Coqui API](https://docs.coqui.ai/docs), however we apply
a few tricks to make it faster and support streaming inference.

### Features
- Voice cloning.
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.
- Streaming inference with < 200ms latency. (See [Streaming inference](#streaming-inference))
- Fine-tuning support. (See [Training](#training))

### Updates with v2
- Improved voice cloning.
- Voices can be cloned with a single audio file or multiple audio files, without any effect on the runtime.
- 2 new languages: Hungarian and Korean.
- Across the board quality improvements.

### Code
Current implementation only supports inference and GPT encoder training.

### Languages
As of now, XTTS-v2 supports 16 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko).

Stay tuned as we continue to add support for more languages. If you have any language requests, please feel free to reach out.

### License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml).

### Contact
Come and join in our 🐸Community. We're active on [Discord](https://discord.gg/fBC58unbKE) and [Twitter](https://twitter.com/coqui_ai).
You can also mail us at info@coqui.ai.

### Inference

#### 🐸TTS Command line

You can check all supported languages with the following command: 

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --list_language_idx
```

You can check all Coqui available speakers with the following command: 

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --list_speaker_idx
```

##### Coqui speakers
You can do inference using one of the available speakers using the following command:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent." \
     --speaker_idx "Ana Florence" \
     --language_idx en \
     --use_cuda true
```

##### Clone a voice
You can clone a speaker voice with a single or multiple references:

###### Single reference

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav \
     --language_idx tr \
     --use_cuda true
```

###### Multiple references
```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav /path/to/target/speaker_2.wav /path/to/target/speaker_3.wav \
     --language_idx tr \
     --use_cuda true
```
or for all wav files in a directory you can use:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/*.wav \
     --language_idx tr \
     --use_cuda true
```

#### 🐸TTS API

##### Clone a voice
You can clone a speaker voice with a single or multiple references:

###### Single reference

Splits the text into sentences and generates audio for each sentence. The audio files are then concatenated to produce the final audio.
You can optionally disable sentence splitting for better coherence but more VRAM and possibly hitting models context length limit.

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["/path/to/target/speaker.wav"],
                language="en",
                split_sentences=True
                )
```

###### Multiple references

You can pass multiple audio files to the `speaker_wav` argument for better voice cloning.

```python
from TTS.api import TTS

# using the default version set in 🐸TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# using a specific version
# 👀 see the branch names for versions on https://huggingface.co/coqui/XTTS-v2/tree/main
# ❗some versions might be incompatible with the API
tts = TTS("xtts_v2.0.2", gpu=True)

# getting the latest XTTS_v2
tts = TTS("xtts", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["/path/to/target/speaker.wav", "/path/to/target/speaker_2.wav", "/path/to/target/speaker_3.wav"],
                language="en")
```

##### Coqui speakers

You can do inference using one of the available speakers using the following code:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker="Ana Florence",
                language="en",
                split_sentences=True
                )
```


#### 🐸TTS Model API

To use the model API, you need to download the model files and pass config and model file paths manually.

#### Manual Inference

If you want to be able to `load_checkpoint` with `use_deepspeed=True` and **enjoy the speedup**, you need to install deepspeed first.

```console
pip install deepspeed==0.10.3
```

##### inference parameters

- `text`: The text to be synthesized.
- `language`: The language of the text to be synthesized.
- `gpt_cond_latent`: The latent vector you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `speaker_embedding`: The speaker embedding you get with get_conditioning_latents. (You can cache for faster inference with same speaker)
- `temperature`: The softmax temperature of the autoregressive model. Defaults to 0.65.
- `length_penalty`: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs. Defaults to 1.0.
- `repetition_penalty`: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.
- `top_k`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 50.
- `top_p`: Lower values mean the decoder produces more "likely" (aka boring) outputs. Defaults to 0.8.
- `speed`: The speed rate of the generated audio. Defaults to 1.0. (can produce artifacts if far from 1.0)
- `enable_text_splitting`: Whether to split the text into sentences and generate audio for each sentence. It allows you to have infinite input length but might loose important context between sentences. Defaults to True.


##### Inference


```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])

print("Inference...")