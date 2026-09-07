
## 🐸Coqui.ai News
- 📣 ⓍTTSv2 is here with 16 languages and better performance across the board.
- 📣 ⓍTTS fine-tuning code is out. Check the [example recipes](https://github.com/coqui-ai/TTS/tree/dev/recipes/ljspeech).
- 📣 ⓍTTS can now stream with <200ms latency.
- 📣 ⓍTTS, our production TTS model that can speak 13 languages, is released [Blog Post](https://coqui.ai/blog/tts/open_xtts), [Demo](https://huggingface.co/spaces/coqui/xtts), [Docs](https://tts.readthedocs.io/en/dev/models/xtts.html)
- 📣 [🐶Bark](https://github.com/suno-ai/bark) is now available for inference with unconstrained voice cloning. [Docs](https://tts.readthedocs.io/en/dev/models/bark.html)
- 📣 You can use [~1100 Fairseq models](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) with 🐸TTS.
- 📣 🐸TTS now supports 🐢Tortoise with faster inference. [Docs](https://tts.readthedocs.io/en/dev/models/tortoise.html)

<div align="center">
<img src="https://static.scarf.sh/a.png?x-pxid=cf317fe7-2188-4721-bc01-124bb5d5dbb2" />

## <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>


**🐸TTS is a library for advanced Text-to-Speech generation.**

🚀 Pretrained models in +1100 languages.

🛠️ Tools for training new models and fine-tuning existing models in any language.

📚 Utilities for dataset analysis and curation.
______________________________________________________________________

[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.gg/5eXr5seRrv)
[![License](<https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg>)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/TTS.svg)](https://badge.fury.io/py/TTS)
[![Covenant](https://camo.githubusercontent.com/7d620efaa3eac1c5b060ece5d6aacfcc8b81a74a04d05cd0398689c01c4463bb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6e7472696275746f72253230436f76656e616e742d76322e3025323061646f707465642d6666363962342e737667)](https://github.com/coqui-ai/TTS/blob/master/CODE_OF_CONDUCT.md)
[![Downloads](https://pepy.tech/badge/tts)](https://pepy.tech/project/tts)
[![DOI](https://zenodo.org/badge/265612440.svg)](https://zenodo.org/badge/latestdoi/265612440)

![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/aux_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/data_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/docker.yaml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/inference_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/style_check.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/text_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/tts_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/vocoder_tests.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests0.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests1.yml/badge.svg)
![GithubActions](https://github.com/coqui-ai/TTS/actions/workflows/zoo_tests2.yml/badge.svg)
[![Docs](<https://readthedocs.org/projects/tts/badge/?version=latest&style=plastic>)](https://tts.readthedocs.io/en/latest/)

</div>

______________________________________________________________________

## 💬 Where to ask questions
Please use our dedicated channels for questions and discussion. Help is much more valuable if it's shared publicly so that more people can benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker]                  |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker]                  |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]                    |
| 🗯 **General Discussion**       | [GitHub Discussions] or [Discord]   |

[github issue tracker]: https://github.com/coqui-ai/tts/issues
[github discussions]: https://github.com/coqui-ai/TTS/discussions
[discord]: https://discord.gg/5eXr5seRrv
[Tutorials and Examples]: https://github.com/coqui-ai/TTS/wiki/TTS-Notebooks-and-Tutorials


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 💼 **Documentation**              | [ReadTheDocs](https://tts.readthedocs.io/en/latest/)
| 💾 **Installation**               | [TTS/README.md](https://github.com/coqui-ai/TTS/tree/dev#installation)|
| 👩‍💻 **Contributing**               | [CONTRIBUTING.md](https://github.com/coqui-ai/TTS/blob/main/CONTRIBUTING.md)|
| 📌 **Road Map**                   | [Main Development Plans](https://github.com/coqui-ai/TTS/issues/378)
| 🚀 **Released Models**            | [TTS Releases](https://github.com/coqui-ai/TTS/releases) and [Experimental Models](https://github.com/coqui-ai/TTS/wiki/Experimental-Released-Models)|
| 📰 **Papers**                    | [TTS Papers](https://github.com/erogol/TTS-papers)|


## 🥇 TTS Performance
<p align="center"><img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/TTS-performance.png" width="800" /></p>

Underlined "TTS*" and "Judy*" are **internal** 🐸TTS models that are not released open-source. They are here to show the potential. Models prefixed with a dot (.Jofish .Abe and .Janice) are real human voices.

## Features
- High-performance Deep Learning models for Text2Speech tasks.
    - Text2Spec models (Tacotron, Tacotron2, Glow-TTS, SpeedySpeech).
    - Speaker Encoder to compute speaker embeddings efficiently.
    - Vocoder models (MelGAN, Multiband-MelGAN, GAN-TTS, ParallelWaveGAN, WaveGrad, WaveRNN)
- Fast and efficient model training.
- Detailed training logs on the terminal and Tensorboard.
- Support for Multi-speaker TTS.
- Efficient, flexible, lightweight but feature complete `Trainer API`.
- Released and ready-to-use models.
- Tools to curate Text2Speech datasets under```dataset_analysis```.
- Utilities to use and test your models.
- Modular (but not too much) code base enabling easy implementation of new ideas.

## Model Implementations
### Spectrogram models
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802)
- Align-TTS: [paper](https://arxiv.org/abs/2003.01950)
- FastPitch: [paper](https://arxiv.org/pdf/2006.06873.pdf)
- FastSpeech: [paper](https://arxiv.org/abs/1905.09263)
- FastSpeech2: [paper](https://arxiv.org/abs/2006.04558)
- SC-GlowTTS: [paper](https://arxiv.org/abs/2104.05557)
- Capacitron: [paper](https://arxiv.org/abs/1906.03402)
- OverFlow: [paper](https://arxiv.org/abs/2211.06892)
- Neural HMM TTS: [paper](https://arxiv.org/abs/2108.13320)
- Delightful TTS: [paper](https://arxiv.org/abs/2110.12612)

### End-to-End Models
- ⓍTTS: [blog](https://coqui.ai/blog/tts/open_xtts)
- VITS: [paper](https://arxiv.org/pdf/2106.06103)
- 🐸 YourTTS: [paper](https://arxiv.org/abs/2112.02418)
- 🐢 Tortoise: [orig. repo](https://github.com/neonbjb/tortoise-tts)
- 🐶 Bark: [orig. repo](https://github.com/suno-ai/bark)

### Attention Methods
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1910.10288)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)
- Dynamic Convolutional Attention: [paper](https://arxiv.org/pdf/1910.10288.pdf)
- Alignment Network: [paper](https://arxiv.org/abs/2108.10447)

### Speaker Encoder
- GE2E: [paper](https://arxiv.org/abs/1710.10467)
- Angular Loss: [paper](https://arxiv.org/pdf/2003.11982.pdf)

### Vocoders
- MelGAN: [paper](https://arxiv.org/abs/1910.06711)
- MultiBandMelGAN: [paper](https://arxiv.org/abs/2005.05106)
- ParallelWaveGAN: [paper](https://arxiv.org/abs/1910.11480)
- GAN-TTS discriminators: [paper](https://arxiv.org/abs/1909.11646)
- WaveRNN: [origin](https://github.com/fatchord/WaveRNN/)
- WaveGrad: [paper](https://arxiv.org/abs/2009.00713)
- HiFiGAN: [paper](https://arxiv.org/abs/2010.05646)
- UnivNet: [paper](https://arxiv.org/abs/2106.07889)

### Voice Conversion
- FreeVC: [paper](https://arxiv.org/abs/2210.15418)

You can also help us implement more models.

## Installation
🐸TTS is tested on Ubuntu 18.04 with **python >= 3.9, < 3.12.**.

If you are only interested in [synthesizing speech](https://tts.readthedocs.io/en/latest/inference.html) with the released 🐸TTS models, installing from PyPI is the easiest option.

```bash
pip install TTS
```

If you plan to code or train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/coqui-ai/TTS
pip install -e .[all,dev,notebooks]  # Select the relevant extras
```

If you are on Ubuntu (Debian), you can also run following commands for installation.

```bash
$ make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
$ make install
```

If you are on Windows, 👑@GuyPaddock wrote installation instructions [here](https://stackoverflow.com/questions/66726331/how-can-i-run-mozilla-tts-coqui-tts-training-with-cuda-on-a-windows-system).


## Docker Image
You can also try TTS without install with the docker image.
Simply run the following command and you will be able to run TTS without installing it.

```bash
docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
python3 TTS/server/server.py --list_models #To get the list of available models
python3 TTS/server/server.py --model_name tts_models/en/vctk/vits # To start a server
```

You can then enjoy the TTS server [here](http://[::1]:5002/)
More details about the docker images (like GPU support) can be found [here](https://tts.readthedocs.io/en/latest/docker_images.html)


## Synthesizing speech by 🐸TTS

### 🐍 Python API

#### Running a multi-speaker and multi-lingual model

```python
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
```

#### Running a single speaker model

```python
# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path=OUTPUT_PATH)

# Example voice cloning with YourTTS in English, French and Portuguese
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")
```

#### Example voice conversion

Converting the voice in `source_wav` to the voice of `target_wav`

```python
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")
```

#### Example voice cloning together with the voice conversion model.
This way, you can clone voices by using any model in 🐸TTS.

```python

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

#### Example text to speech using **Fairseq models in ~1100 languages** 🤯.
For Fairseq models, use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
You can find the language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html)
and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

```python
# TTS with on the fly voice conversion
api = TTS("tts_models/deu/fairseq/vits")
api.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)
```

### Command-line `tts`

<!-- begin-tts-readme -->

Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

#### Single Speaker Models

- List provided models:

  ```
  $ tts --list_models
  ```

- Get model info (for both tts_models and vocoder_models):

  - Query by type/name:
    The model_info_by_name uses the name as it from the --list_models.
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```
    For example:
    ```
    $ tts --model_info_by_name tts_models/tr/common-voice/glow-tts
    $ tts --model_info_by_name vocoder_models/en/ljspeech/hifigan_v2
    ```
  - Query by type/idx:
    The model_query_idx uses the corresponding idx from --list_models.

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

    For example:

    ```
    $ tts --model_info_by_idx tts_models/3
    ```

  - Query info for model info by full name:
    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav
  ```

- Run TTS and pipe out the generated TTS wav file data:

  ```
  $ tts --text "Text for TTS" --pipe_out --out_path output/path/speech.wav | aplay
  ```

- Run a TTS model with its default vocoder model:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --out_path output/path/speech.wav
  ```

- Run with specific TTS and vocoder models from the list:

  ```
  $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --out_path output/path/speech.wav
  ```

  For example:

  ```
  $ tts --text "Text for TTS" --model_name "tts_models/en/ljspeech/glow-tts" --vocoder_name "vocoder_models/en/ljspeech/univnet" --out_path output/path/speech.wav
  ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
  ```

- Run your own TTS and Vocoder models:

  ```
  $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
      --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
  ```

#### Multi-speaker Models

- List the available speakers and choose a <speaker_id> among them:

  ```
  $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
  ```

- Run the multi-speaker TTS model with the target speaker ID:

  ```
  $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
  ```

- Run your own multi-speaker TTS model:

  ```
  $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/model.pth --config_path path/to/config.json --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
  ```

### Voice Conversion Models

```
$ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
```

<!-- end-tts-readme -->

## Directory Structure
```
|- notebooks/       (Jupyter Notebooks for model evaluation, parameter selection and data analysis.)
|- utils/           (common utilities.)
|- TTS
    |- bin/             (folder for all the executables.)
      |- train*.py                  (train your target model.)
      |- ...
    |- tts/             (text to speech models)
        |- layers/          (model layer definitions)
        |- models/          (model definitions)
        |- utils/           (model specific utilities.)
    |- speaker_encoder/ (Speaker Encoder models.)
        |- (same)
    |- vocoder/         (Vocoder models.)
        |- (same)
```


## 🌐 Web Resources & Interactive Index
- [FROM NERD TO POPULAR BALLERINA CAPPUCCINA](https://welearnaction.onrender.com/from-nerd-to-popular-ballerina-cappuccina.html)
- [HEX SENSE](https://eduquests.github.io/hex-sense.html)
- [CATEGORY 2D1 070](https://learnaction.netlify.app/category-2d1-070.html)
- [INDEX19](https://learnaction.github.io/index19.html)
- [CATEGORY MONSTER206](https://learnaction.github.io/category-monster206.html)
- [CATEGORY AGILITY](https://learnaction.netlify.app/category-agility.html)
- [SPACEIO](https://welearnaction.onrender.com/spaceio.html)
- [FREDDYS NIGHTMARES RETURN HORROR NEW YEAR](https://learnaction.netlify.app/freddys-nightmares-return-horror-new-year.html)
- [MERGEST KINGDOM](https://welearnaction.onrender.com/mergest-kingdom.html)
- [BATTLETABS](https://learnaction.github.io/battletabs.html)
- [CATEGORY STICKMAN175](https://learnaction.github.io/category-stickman175.html)
- [SITEMAP](https://cryptotify.netlify.app/sitemap.html)
- [SHAPE SHIFTING](https://welearnaction.onrender.com/shape-shifting.html)
- [CATEGORY SNAKE40](https://learnaction.github.io/category-snake40.html)
- [SITEMAP](https://skillcrafts.github.io/sitemap.html)
- [WOODS OF NEVIA FOREST SURVIVAL](https://welearnaction.onrender.com/woods-of-nevia-forest-survival.html)
- [BOTTLE LOGIC](https://learnaction.github.io/bottle-logic.html)
- [CATEGORY PUZZLE 5](https://learnaction.github.io/category-puzzle-5.html)
- [MARBLE BLAST](https://welearnaction.onrender.com/marble-blast.html)
- [TIKTOK TRENDS COLORED DENIM](https://welearnaction.onrender.com/tiktok-trends-colored-denim.html)
- [SNOW RACE 3D FUN RACING](https://learnaction.github.io/snow-race-3d-fun-racing.html)
- [STUNT MULTIPLAYER ARENA](https://welearnaction.onrender.com/stunt-multiplayer-arena.html)
- [HOME ISLAND](https://learnaction.github.io/home-island.html)
- [SITEMAP](https://quizverses.pages.dev/sitemap.html)
- [CATEGORY ARENA255](https://learnaction.netlify.app/category-arena255.html)
- [HORSE CHAMPS](https://learnaction.netlify.app/horse-champs.html)
- [CATEGORY FOOTBALL](https://eduquests.netlify.app/category-football.html)
- [CATEGORY RACING DRIVING 2](https://learnaction.netlify.app/category-racing-driving-2.html)
- [STICKMAN RAGDOLL PLAYGROUND](https://welearnaction.onrender.com/stickman-ragdoll-playground.html)
- [CATEGORY ROGUELIKE38](https://learnaction.github.io/category-roguelike38.html)
- [TERMS](https://cryptotify.github.io/terms.html)
- [BRAWLER MAN FIST OF FURY](https://learnaction.netlify.app/brawler-man-fist-of-fury.html)
- [CATEGORY BLOCK91](https://learnaction.github.io/category-block91.html)
- [SITEMAP](https://themindzone.pages.dev/sitemap.html)
- [CANDY POP MANIA](https://learnaction.netlify.app/candy-pop-mania.html)
- [PLANET MERGE](https://welearnaction.onrender.com/planet-merge.html)
- [CATEGORY SIMULATION](https://learnaction.netlify.app/category-simulation.html)
- [CATEGORY 3D1 371](https://learnaction.netlify.app/category-3d1-371.html)
- [BRAINROT HOLE](https://learnaction.netlify.app/brainrot-hole.html)
- [TERRA CRAFT WORLD](https://learnaction.github.io/terra-craft-world.html)
- [SWEET DESSERT HOLE](https://eduquests.netlify.app/sweet-dessert-hole.html)
- [BUBBLE SHOOTER REMASTERED](https://learnaction.github.io/bubble-shooter-remastered.html)
- [TERMS](https://ilearnworldjp.pages.dev/terms.html)
- [SQUAD ASSEMBLER](https://learnaction.netlify.app/squad-assembler.html)
- [SPACE SHOOTER SPEED TYPING CHALLENGE](https://welearnaction.onrender.com/space-shooter-speed-typing-challenge.html)
- [BUNNIES SORT](https://learnaction.netlify.app/bunnies-sort.html)
- [SUPER NINJA BALLOON](https://welearnaction.onrender.com/super-ninja-balloon.html)
- [SKIBRONX RUNNER](https://welearnaction.onrender.com/skibronx-runner.html)
- [FIND THE CAT CAT SEARCH](https://welearnaction.onrender.com/find-the-cat-cat-search.html)
- [PRIVACY](https://skillplay.github.io/privacy.html)
- [CONTACT](https://learnaction.netlify.app/contact.html)
- [HIDDEN OBJECT ROOMS EXPLORATION](https://welearnaction.onrender.com/hidden-object-rooms-exploration.html)
- [CATEGORY ADVENTURE 3](https://learnaction.github.io/category-adventure-3.html)
- [CONTACT](https://eduquests.netlify.app/contact.html)
- [INDEX17](https://eduquests.netlify.app/index17.html)
- [TERMS](https://welearnaction.onrender.com/terms.html)
- [TRAFFIC RUN PUZZLE](https://welearnaction.onrender.com/traffic-run-puzzle.html)
- [CARJAMCOLOR](https://welearnaction.onrender.com/carjamcolor.html)
- [HIDE BALL](https://welearnaction.onrender.com/hide-ball.html)
- [FAR ORION NEW WORLDS](https://welearnaction.onrender.com/far-orion-new-worlds.html)
- [EGG DASH](https://learnaction.netlify.app/egg-dash.html)
- [CATEGORY SNIPER39](https://learnaction.netlify.app/category-sniper39.html)
- [STICKMAN IN SPACE](https://learnaction.netlify.app/stickman-in-space.html)
- [CATEGORY BASKETBALL](https://eduquests.netlify.app/category-basketball.html)
- [ANGRY CITY SMASHER](https://learnaction.github.io/angry-city-smasher.html)
- [CAR COLLISION MASTER](https://learnaction.github.io/car-collision-master.html)
- [MUSHROOM BLOCKS](https://learnaction.github.io/mushroom-blocks.html)
- [LATUTU HOLIDAY GIFT HUNT](https://learnaction.netlify.app/latutu-holiday-gift-hunt.html)
- [SPELLMIND](https://learnaction.github.io/spellmind.html)
- [ITALIAN BRAINROT CHALLENGE](https://learnaction.github.io/italian-brainrot-challenge.html)
- [INDEX8](https://eduquests.netlify.app/index8.html)
- [LABUBU COLORING ADVENTURE](https://learnaction.github.io/labubu-coloring-adventure.html)
- [ONLINE PORTAL](https://ilearnworldkr.pages.dev/)
- [RIFT OF HELL DEMONS WAR](https://learnaction.github.io/rift-of-hell-demons-war.html)
- [PET TILE MASTER](https://welearnaction.onrender.com/pet-tile-master.html)
- [BALL ROLLING SLOPE](https://learnaction.github.io/ball-rolling-slope.html)
- [BATTLEDUDES IO](https://learnaction.github.io/battledudes-io.html)
- [ONLINE PORTAL](https://eduquests.netlify.app/)
- [INDEX2](https://learnaction.github.io/index2.html)
- [MERGEST KINGDOM](https://learnaction.netlify.app/mergest-kingdom.html)
- [WORM ESCAPE](https://welearnaction.onrender.com/worm-escape.html)
- [INDEX2](https://eduquests.netlify.app/index2.html)
- [INDEX20](https://eduquests.netlify.app/index20.html)
- [WORM ESCAPE](https://learnaction.github.io/worm-escape.html)
- [CATEGORY ANIMAL216](https://eduquests.netlify.app/category-animal216.html)
- [OUTSIDE](https://learnaction.github.io/outside.html)
- [CATEGORY BALL173](https://learnaction.netlify.app/category-ball173.html)
- [BASKET CHAMPS](https://learnaction.netlify.app/basket-champs.html)
- [COLOR DOTS CHALLENGE](https://welearnaction.onrender.com/color-dots-challenge.html)
- [CODE RUNNER BINARY CONFUSION](https://learnaction.github.io/code-runner-binary-confusion.html)
- [8 BALL POOL BILLIARDS MULTIPLAYER](https://learnaction.netlify.app/8-ball-pool-billiards-multiplayer.html)
- [RABBIT CARROT](https://learnaction.netlify.app/rabbit-carrot.html)
- [RACING BALL ADVENTURE](https://learnaction.netlify.app/racing-ball-adventure.html)
- [RIDDLEMATH](https://learnaction.github.io/riddlemath.html)
- [TRANSFORMERS BATTLE FOR THE CITY](https://welearnaction.onrender.com/transformers-battle-for-the-city.html)
- [INDEX21](https://welearnaction.onrender.com/index21.html)
- [CATEGORY AGILITY 2](https://eduquests.netlify.app/category-agility-2.html)
- [MUKI WIZARD](https://learnaction.netlify.app/muki-wizard.html)
- [VEX 9](https://learnaction.github.io/vex-9.html)
- [INDEX3](https://eduquests.netlify.app/index3.html)
- [TOCO TEENS HALLOWEEN PARTY](https://learnaction.github.io/toco-teens-halloween-party.html)
- [CATEGORY 1 PLAYER139](https://eduquests.netlify.app/category-1-player139.html)
- [GOO SLIME JUMP](https://welearnaction.onrender.com/goo-slime-jump.html)
- [BLACK HOLE BEAUTY MAKEUP](https://ieduquests.web.app/black-hole-beauty-makeup.html)
- [PRIVACY](https://brainquests.onrender.com/privacy.html)
- [DINO SLIDE](https://eduquests.netlify.app/dino-slide.html)
- [SERIOUS HEAD](https://eduquests.onrender.com/serious-head.html)
- [CATEGORY CASUAL](https://learnaction.github.io/category-casual.html)
- [TOY RUMBLE 3D](https://eduquests.onrender.com/toy-rumble-3d.html)
- [BUILD A GO KART](https://learnaction.github.io/build-a-go-kart.html)
- [PRESS A TO PARTY](https://eduquests.onrender.com/press-a-to-party.html)
- [VIBE COLOURING](https://learnaction.github.io/vibe-colouring.html)
- [MERGE HOME MANIA](https://learnaction.github.io/merge-home-mania.html)
- [PUSH THE FROG](https://learnaction.github.io/push-the-frog.html)
- [SAFE MERGE](https://learnaction.github.io/safe-merge.html)
- [REAL GT RACING SIMULATOR](https://welearnaction.onrender.com/real-gt-racing-simulator.html)
- [PEOPLE PLAYGROUND RAGDOLL ARENA](https://eduquests.netlify.app/people-playground-ragdoll-arena.html)
- [INDEX18](https://eduquests.netlify.app/index18.html)
- [CATEGORY BOOKMARKLET](https://eduquests.github.io/category-bookmarklet.html)
- [POWER PUZZLE](https://learnaction.github.io/power-puzzle.html)
- [CATEGORY ART](https://learnaction.netlify.app/category-art.html)
- [UNBLOCK IT 3D](https://eduquests.onrender.com/unblock-it-3d.html)
- [STICKMAN PRISON AND LOVE](https://welearnaction.onrender.com/stickman-prison-and-love.html)
- [CATEGORY SHOOTER](https://eduquests.github.io/category-shooter.html)
- [SHOT CAN WILD](https://learnaction.netlify.app/shot-can-wild.html)
- [MAGIC BUBBLES](https://eduquests.github.io/magic-bubbles.html)
- [CUBE COMBO](https://eduquests.github.io/cube-combo.html)
- [SHOP SORTING 2](https://eduquests.github.io/shop-sorting-2.html)
- [SNOW BALL RACING MUTLIPLAYER](https://learnaction.github.io/snow-ball-racing-mutliplayer.html)
- [CATEGORY BOOKMARK](https://eduquests.github.io/category-bookmark.html)
