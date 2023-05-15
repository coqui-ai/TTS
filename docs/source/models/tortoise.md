# Tortoise 🐢
Tortoise is a very expressive TTS system with impressive voice cloning capabilities. It is based on an GPT like autogressive acoustic model that converts input
text to discritized acouistic tokens, a diffusion model that converts these tokens to melspeectrogram frames and a Univnet vocoder to convert the spectrograms to
the final audio signal. The important downside is that Tortoise is very slow compared to the parallel TTS models like VITS. 

Big thanks to 👑[@manmay-nakhashi](https://github.com/manmay-nakhashi) who helped us implement Tortoise in 🐸TTS.

Example use:

```python
from TTS.tts.configs.tortoise_config import TortoiseConfig
from TTS.tts.models.tortoise import Tortoise

config = TortoiseConfig()
model = Tortoise.inif_from_config(config)
model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)

# with random speaker
output_dict = model.synthesize(text, config, speaker_id="random", extra_voice_dirs=None, **kwargs)

# cloning a speaker
output_dict = model.synthesize(text, config, speaker_id="speaker_n", extra_voice_dirs="path/to/speaker_n/", **kwargs)
```

Using 🐸TTS API:

```python
from TTS.api import TTS
tts = TTS("tts_models/en/multi-dataset/tortoise-v2")

# cloning `lj` voice from `TTS/tts/utils/assets/tortoise/voices/lj`
# with custom inference settings overriding defaults. 
tts.tts_to_file(text="Hello, my name is Manmay , how are you?", 
                file_path="output.wav",
                voice_dir="TTS/tts/utils/assets/tortoise/voices/",
                speaker="lj",
                num_autoregressive_samples=1,
                diffusion_iterations=10)

# Using presets with the same voice
tts.tts_to_file(text="Hello, my name is Manmay , how are you?", 
                file_path="output.wav",
                voice_dir="TTS/tts/utils/assets/tortoise/voices/",
                speaker="lj",
                preset="ultra_fast")

# Random voice generation
tts.tts_to_file(text="Hello, my name is Manmay , how are you?", 
                file_path="output.wav")
```

Using 🐸TTS Command line:

```console
# cloning the `lj` voice
tts --model_name  tts_models/en/multi-dataset/tortoise-v2 \
--text "This is an example." \ 
--out_path "/data/speech_synth/coqui-tts/TTS/tests/outputs/output.wav" \
--voice_dir TTS/tts/utils/assets/tortoise/voices/ \
--speaker_idx "lj" \
--progress_bar True

# Random voice generation
tts --model_name  tts_models/en/multi-dataset/tortoise-v2 \
--text "This is an example." \
--out_path "/data/speech_synth/coqui-tts/TTS/tests/outputs/output.wav" \
--progress_bar True
```


## Important resources & papers
- Original Repo: https://github.com/neonbjb/tortoise-tts
- Faster implementation: https://github.com/152334H/tortoise-tts-fast
- Univnet: https://arxiv.org/abs/2106.07889
- Latent Diffusion:https://arxiv.org/abs/2112.10752
- DALL-E: https://arxiv.org/abs/2102.12092

## TortoiseConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.tortoise_config.TortoiseConfig
    :members:
```

## TortoiseArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.tortoise.TortoiseArgs
    :members:
```

## Tortoise Model
```{eval-rst}
.. autoclass:: TTS.tts.models.tortoise.Tortoise
    :members:
```
