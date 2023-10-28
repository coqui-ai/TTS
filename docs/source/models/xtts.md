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
As of now, XTTS-v1.1 supports 14 languages: English, Spanish, French, German, Italian, Portuguese,
Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese (Simplified) and Japanese.

Stay tuned as we continue to add support for more languages. If you have any language requests, please feel free to reach out.

### License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml).

### Contact
Come and join in our 🐸Community. We're active on [Discord](https://discord.gg/fBC58unbKE) and [Twitter](https://twitter.com/coqui_ai).
You can also mail us at info@coqui.ai.

### Inference
#### 🐸TTS API

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1.1", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")
```

#### 🐸TTS Command line

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v1.1 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav \
     --language_idx tr \
     --use_cuda true
```

#### model directly

If you want to be able to run with `use_deepspeed=True` and enjoy the speedup, you need to install deepspeed first.

```console
pip install deepspeed==0.8.3
```

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
gpt_cond_latent, diffusion_conditioning, speaker_embedding = model.get_conditioning_latents(audio_path="reference.wav")

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    diffusion_conditioning,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
```


#### streaming inference

Here the goal is to stream the audio as it is being generated. This is useful for real-time applications.
Streaming inference is typically slower than regular inference, but it allows to get a first chunk of audio faster.


```python
import os
import time
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
gpt_cond_latent, _, speaker_embedding = model.get_conditioning_latents(audio_path="reference.wav")

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding
)
    
wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
```


### Training

A recipe for `XTTS_v1.1` GPT encoder training using `LJSpeech` dataset is available at https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech/xtts_v1/train_gpt_xtts.py

You need to change the fields of the `BaseDatasetConfig` to match your dataset and then update `GPTArgs` and `GPTTrainerConfig` fields as you need. By default, it will use the same parameters that XTTS v1.1 model was trained with. To speed up the model convergence, as default, it will also download the XTTS v1.1 checkpoint and load it.

After training you can do inference following the code bellow.

```python
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
CONFIG_PATH = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT-October-23-2023_10+36AM-653f2e75/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "recipes/ljspeech/xtts_v1/run/training/XTTS_v1.1_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "recipes/ljspeech/xtts_v1/run/training/GPT_XTTS_LJSpeech_FT/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "LjSpeech_reference.wav"

# output wav path
OUTPUT_WAV_PATH = "xtts-ft.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, diffusion_conditioning, speaker_embedding = model.get_conditioning_latents(audio_path=SPEAKER_REFERENCE)

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    diffusion_conditioning,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
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
