# ⓍTTS
ⓍTTS is a super cool Text-to-Speech model that lets you clone voices in different languages by using just a quick 3-second audio clip. Built on the 🐢Tortoise,
ⓍTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy.
There is no need for an excessive amount of training data that spans countless hours.

This is the same model that powers [Coqui Studio](https://coqui.ai/), and [Coqui API](https://docs.coqui.ai/docs), however we apply
a few tricks to make it faster and support streaming inference.

### Features
- Voice cloning with just a 3-second audio clip.
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.

### Code
Current implementation only supports inference.

### Languages
As of now, XTTS-v1 supports 13 languages: English, Spanish, French, German, Italian, Portuguese,
Polish, Turkish, Russian, Dutch, Czech, Arabic, and Chinese (Simplified).

Stay tuned as we continue to add support for more languages. If you have any language requests, please feel free to reach out.

### License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml).

### Contact
Come and join in our 🐸Community. We're active on [Discord](https://discord.gg/fBC58unbKE) and [Twitter](https://twitter.com/coqui_ai).
You can also mail us at info@coqui.ai.

Using 🐸TTS API:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")

# generate speech by cloning a voice using custom settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en",
                decoder_iterations=30)
```

Using 🐸TTS Command line:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v1 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav \
     --language_idx tr \
     --use_cuda true
```

Using model directly:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/data/TTS-public/_refclips/3.wav",
    gpt_cond_len=3,
    language="en",
)
```


## Important resources & papers
- VallE: https://arxiv.org/abs/2301.02111
- Tortoise Repo: https://github.com/neonbjb/tortoise-tts
- Faster implementation: https://github.com/152334H/tortoise-tts-fast
- Univnet: https://arxiv.org/abs/2106.07889
- Latent Diffusion:https://arxiv.org/abs/2112.10752
- DALL-E: https://arxiv.org/abs/2102.12092


## XttsConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.xtts_config.XttsConfig
    :members:
```

## XttsArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.xtts.XttsArgs
    :members:
```

## XTTS Model
```{eval-rst}
.. autoclass:: TTS.tts.models.xtts.XTTS
    :members:
```
