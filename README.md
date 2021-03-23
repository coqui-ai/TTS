# <img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/coqui-log-green-TTS.png" height="56"/>

🐸TTS is a library for advanced Text-to-Speech generation. It's built on the latest research, was designed to achieve the best trade-off among ease-of-training, speed and quality.
🐸TTS comes with [pretrained models](https://github.com/coqui-ai/TTS/wiki/Released-Models), tools for measuring dataset quality and already used in **20+ languages** for products and research projects.

[![CircleCI](https://github.com/coqui-ai/TTS/actions/workflows/main.yml/badge.svg)]()
[![License](<https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg>)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/TTS.svg)](https://badge.fury.io/py/TTS)
[![Covenant](https://camo.githubusercontent.com/7d620efaa3eac1c5b060ece5d6aacfcc8b81a74a04d05cd0398689c01c4463bb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6e7472696275746f72253230436f76656e616e742d76322e3025323061646f707465642d6666363962342e737667)](https://github.com/coqui-ai/TTS/blob/master/CODE_OF_CONDUCT.md)
[![Gitter](https://badges.gitter.im/coqui-ai/TTS.svg)](https://gitter.im/coqui-ai/TTS?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

📰 [**Subscribe to 🐸Coqui.ai Newsletter**](https://coqui.ai/?subscription=true)

📢 [English Voice Samples](https://erogol.github.io/ddc-samples/) and [SoundCloud playlist](https://soundcloud.com/user-565970875/pocket-article-wavernn-and-tacotron2)

👩🏽‍🍳 [TTS training recipes](https://github.com/erogol/TTS_recipes)

📄 [Text-to-Speech paper collection](https://github.com/erogol/TTS-papers)

## 💬 Where to ask questions
Please use our dedicated channels for questions and discussion. Help is much more valuable if it's shared publicly, so that more people can benefit from it.

| Type                            | Platforms                               |
| ------------------------------- | --------------------------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker]                  |
| ❔ **FAQ**                      | [TTS/Wiki](https://github.com/coqui-ai/TTS/wiki/FAQ)                              |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker]                  |
| 👩‍💻 **Usage Questions**          | [Github Discussions]                    |
| 🗯 **General Discussion**       | [Github Discussions] or [Gitter Room] or [Matrix Room]  |

[github issue tracker]: https://github.com/coqui-ai/tts/issues
[github discussions]: https://github.com/coqui-ai/TTS/discussions
[gitter room]: https://gitter.im/coqui-ai/TTS?utm_source=share-link&utm_medium=link&utm_campaign=share-link
[matrix room]: https://matrix.to/#/!KTePhNahjgiVumkqca:matrix.org?via=matrix.org&via=bethselamin.de&via=lossy.network
[Tutorials and Examples]: https://github.com/coqui-ai/TTS/wiki/TTS-Notebooks-and-Tutorials


## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 💾 **Installation** | [TTS/README.md](https://github.com/coqui-ai/TTS/tree/dev#install-tts)|
| 👩🏾‍🏫 **Tutorials and Examples**  | [TTS/Wiki](https://github.com/coqui-ai/TTS/wiki/%F0%9F%90%B8-TTS-Notebooks,-Examples-and-Tutorials) |
| 🚀 **Released Models**         | [TTS Releases](https://github.com/coqui-ai/TTS/releases) and [Experimental Models](https://github.com/coqui-ai/TTS/wiki/Experimental-Released-Models)|
| 💻 **Docker Image**            | [Repository by @synesthesiam](https://github.com/synesthesiam/docker-coqui-aitts)|
| 🖥️ **Demo Server**             | [TTS/server](https://github.com/coqui-ai/TTS/tree/master/TTS/server)|
| 🤖 **Synthesize speech** | [TTS/README.md](https://github.com/coqui-ai/TTS#example-synthesizing-speech-on-terminal-using-the-released-models)|

## 🥇 TTS Performance
<p align="center"><img src="https://raw.githubusercontent.com/coqui-ai/TTS/main/images/TTS-performance.png" width="800" /></p>

Underlined "TTS*" and "Judy*" are 🐸TTS models
<!-- [Details...](https://github.com/coqui-ai/TTS/wiki/Mean-Opinion-Score-Results) -->

## Features
- High performance Deep Learning models for Text2Speech tasks.
    - Text2Spec models (Tacotron, Tacotron2, Glow-TTS, SpeedySpeech).
    - Speaker Encoder to compute speaker embeddings efficiently.
    - Vocoder models (MelGAN, Multiband-MelGAN, GAN-TTS, ParallelWaveGAN, WaveGrad, WaveRNN)
- Fast and efficient model training.
- Detailed training logs on console and Tensorboard.
- Support for multi-speaker TTS.
- Efficient Multi-GPUs training.
- Ability to convert PyTorch models to Tensorflow 2.0 and TFLite for inference.
- Released models in PyTorch, Tensorflow and TFLite.
- Tools to curate Text2Speech datasets under```dataset_analysis```.
- Demo server for model testing.
- Notebooks for extensive model benchmarking.
- Modular (but not too much) code base enabling easy testing for new ideas.

## Implemented Models
### Text-to-Spectrogram
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)
- Speedy-Speech: [paper](https://arxiv.org/abs/2008.03802)

### Attention Methods
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1907.09006)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)

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

You can also help us implement more models. Some 🐸TTS related work can be found [here](https://github.com/erogol/TTS-papers).

## Install TTS
🐸TTS is tested on Ubuntu 18.04 with **python >= 3.6, < 3.9**.

If you are only interested in [synthesizing speech](https://github.com/coqui-ai/TTS/tree/dev#example-synthesizing-speech-on-terminal-using-the-released-models) with the released 🐸TTS models, installing from PyPI is the easiest option.

```bash
pip install TTS
```

If you plan to code or train models, clone 🐸TTS and install it locally.

```bash
git clone https://github.com/coqui-ai/TTS
pip install -e .
```

We use ```espeak-ng``` to convert graphemes to phonemes. You might need to install separately.
```bash
sudo apt-get install espeak-ng
```

## Directory Structure
```
|- notebooks/       (Jupyter Notebooks for model evaluation, parameter selection and data analysis.)
|- utils/           (common utilities.)
|- TTS
    |- bin/             (folder for all the executables.)
      |- train*.py                  (train your target model.)
      |- distribute.py              (train your TTS model using Multiple GPUs.)
      |- compute_statistics.py      (compute dataset statistics for normalization.)
      |- convert*.py                (convert target torch model to TF.)
    |- tts/             (text to speech models)
        |- layers/          (model layer definitions)
        |- models/          (model definitions)
        |- tf/              (Tensorflow 2 utilities and model implementations)
        |- utils/           (model specific utilities.)
    |- speaker_encoder/ (Speaker Encoder models.)
        |- (same)
    |- vocoder/         (Vocoder models.)
        |- (same)
```

## Sample Model Output
Below you see Tacotron model state after 16K iterations with batch-size 32 with LJSpeech dataset.

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio examples: [soundcloud](https://soundcloud.com/user-565970875/pocket-article-wavernn-and-tacotron2)

<img src="images/example_model_output.png?raw=true" alt="example_output" width="400"/>

## Datasets and Data-Loading
🐸TTS provides a generic dataloader easy to use for your custom dataset.
You just need to write a simple function to format the dataset. Check ```datasets/preprocess.py``` to see some examples.
After that, you need to set ```dataset``` fields in ```config.json```.

Some of the public datasets that we successfully applied 🐸TTS:

- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- [Nancy](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
- [TWEB](https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset)
- [M-AI-Labs](http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
- [LibriTTS](https://openslr.org/60/)
- [Spanish](https://drive.google.com/file/d/1Sm_zyBo67XHkiFhcRSQ4YaHPYM0slO_e/view?usp=sharing) - thx! @carlfm01

## Example: Synthesizing Speech on Terminal Using the Released Models.

After the installation, 🐸TTS provides a CLI interface for synthesizing speech using pre-trained models. You can either use your own model or the release models under 🐸TTS.

Listing released 🐸TTS models.
```bash
tts --list_models
```

Run a tts and a vocoder model from the released model list. (Simply copy and paste the full model names from the list as arguments for the command below.)
```bash
tts --text "Text for TTS" \
    --model_name "<type>/<language>/<dataset>/<model_name>" \
    --vocoder_name "<type>/<language>/<dataset>/<model_name>" \
    --out_path folder/to/save/output/
```

Run your own TTS model (Using Griffin-Lim Vocoder)
```bash
tts --text "Text for TTS" \
    --model_path path/to/model.pth.tar \
    --config_path path/to/config.json \
    --out_path output/path/speech.wav
```

Run your own TTS and Vocoder models
```bash
tts --text "Text for TTS" \
    --model_path path/to/config.json \
    --config_path path/to/model.pth.tar \
    --out_path output/path/speech.wav \
    --vocoder_path path/to/vocoder.pth.tar \
    --vocoder_config_path path/to/vocoder_config.json
```

**Note:** You can use ```./TTS/bin/synthesize.py``` if you prefer running ```tts``` from the TTS project folder.

## Example: Training and Fine-tuning LJ-Speech Dataset
Here you can find a [CoLab](https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b) notebook for a hands-on example, training LJSpeech. Or you can manually follow the guideline below.

To start with, split ```metadata.csv``` into train and validation subsets respectively ```metadata_train.csv``` and ```metadata_val.csv```. Note that for text-to-speech, validation performance might be misleading since the loss value does not directly measure the voice quality to the human ear and it also does not measure the attention module performance. Therefore, running the model with new sentences and listening to the results is the best way to go.

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 1100 metadata_shuf.csv > metadata_val.csv
```

To train a new model, you need to define your own ```config.json``` to define model details, trainin configuration and more (check the examples). Then call the corressponding train script.

For instance, in order to train a tacotron or tacotron2 model on LJSpeech dataset, follow these steps.

```bash
python TTS/bin/train_tacotron.py --config_path TTS/tts/configs/config.json
```

To fine-tune a model, use ```--restore_path```.

```bash
python TTS/bin/train_tacotron.py --config_path TTS/tts/configs/config.json --restore_path /path/to/your/model.pth.tar
```

To continue an old training run, use ```--continue_path```.

```bash
python TTS/bin/train_tacotron.py --continue_path /path/to/your/run_folder/
```

For multi-GPU training, call ```distribute.py```. It runs any provided train script in multi-GPU setting.

```bash
CUDA_VISIBLE_DEVICES="0,1,4" python TTS/bin/distribute.py --script train_tacotron.py --config_path TTS/tts/configs/config.json
```

Each run creates a new output folder accomodating used ```config.json```, model checkpoints and tensorboard logs.

In case of any error or intercepted execution, if there is no checkpoint yet under the output folder, the whole folder is going to be removed.

You can also enjoy Tensorboard,  if you point Tensorboard argument```--logdir``` to the experiment folder.

## Contribution guidelines
Please follow the steps below as you send a PR to 🐸. It helps us to keep things organized.

1. Create a new branch.
2. Implement your changes.
3. (if applicable) Add [Google Style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
4. (if applicable) Implement a test case under ```tests``` folder.
5. (Optional but Prefered) Run tests.
```bash
./run_tests.sh
```
6. Run the ```pylint``` linter.
```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```
7. Send a PR to ```dev``` branch, explain what the change is about.
8. Let us discuss until we make it perfect :) 💪.
9. We merge it to the ```dev``` branch once things look good.

Feel free to ping us at any step you need help using our communication channels.
[Here](https://github.com/firstcontributions/first-contributions) is a good resource for complete beginners.


### Acknowledgement
- https://github.com/keithito/tacotron (Dataset pre-processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
- https://github.com/kan-bayashi/ParallelWaveGAN (vocoder library)
- https://github.com/jaywalnut310/glow-tts (Original Glow-TTS implementation)
- https://github.com/fatchord/WaveRNN/ (Original WaveRNN implementation)
