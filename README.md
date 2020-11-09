<p align="center"><img src="https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png" data-canonical-src="![TTS banner](https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png =250x250)
" width="320" height="95" /></p>

<br/>

<p align='center'>
    <img src="https://travis-ci.org/mozilla/TTS.svg?branch=dev"/>
    <a href='https://discourse.mozilla.org/c/tts'><img src="https://img.shields.io/badge/discourse-online-green.svg"/></a>
    <a href='https://opensource.org/licenses/MPL-2.0'> <img src="https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg"/></a>
</p>

<br/>

Mozilla TTS is a deep learning based Text2Speech project, low in cost and high in quality.

This project is a part of [Mozilla Common Voice](https://voice.mozilla.org/en).

English Voice Samples: https://erogol.github.io/ddc-samples/

TTS training recipes: https://github.com/erogol/TTS_recipes

TTS paper collection: https://github.com/erogol/TTS-papers

[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/0)](https://sourcerer.io/fame/erogol/erogol/TTS/links/0)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/1)](https://sourcerer.io/fame/erogol/erogol/TTS/links/1)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/2)](https://sourcerer.io/fame/erogol/erogol/TTS/links/2)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/3)](https://sourcerer.io/fame/erogol/erogol/TTS/links/3)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/4)](https://sourcerer.io/fame/erogol/erogol/TTS/links/4)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/5)](https://sourcerer.io/fame/erogol/erogol/TTS/links/5)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/6)](https://sourcerer.io/fame/erogol/erogol/TTS/links/6)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/7)](https://sourcerer.io/fame/erogol/erogol/TTS/links/7)

## TTS Performance
<p align="center"><img src="https://discourse-prod-uploads-81679984178418.s3.dualstack.us-west-2.amazonaws.com/optimized/3X/6/4/6428f980e9ec751c248e591460895f7881aec0c6_2_1035x591.png" width="800" /></p>

"Mozilla*" and "Judy*" are our models.
[Details...](https://github.com/mozilla/TTS/wiki/Mean-Opinion-Score-Results)

## Provided Models and Methods
Text-to-Spectrogram:
- Tacotron: [paper](https://arxiv.org/abs/1703.10135)
- Tacotron2: [paper](https://arxiv.org/abs/1712.05884)
- Glow-TTS: [paper](https://arxiv.org/abs/2005.11129)

Attention Methods:
- Guided Attention: [paper](https://arxiv.org/abs/1710.08969)
- Forward Backward Decoding: [paper](https://arxiv.org/abs/1907.09006)
- Graves Attention: [paper](https://arxiv.org/abs/1907.09006)
- Double Decoder Consistency: [blog](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/)

Speaker Encoder:
- GE2E: [paper](https://arxiv.org/abs/1710.10467)

Vocoders:
- MelGAN: [paper](https://arxiv.org/abs/1710.10467)
- MultiBandMelGAN: [paper](https://arxiv.org/abs/2005.05106)
- ParallelWaveGAN: [paper](https://arxiv.org/abs/1910.11480)
- GAN-TTS discriminators: [paper](https://arxiv.org/abs/1909.11646)
- WaveRNN: [origin][https://github.com/fatchord/WaveRNN/]
- WaveGrad: [paper][https://arxiv.org/abs/2009.00713]

You can also help us implement more models. Some TTS related work can be found [here](https://github.com/erogol/TTS-papers).

## Features
- High performance Deep Learning models for Text2Speech tasks.
    - Text2Spec models (Tacotron, Tacotron2).
    - Speaker Encoder to compute speaker embeddings efficiently.
    - Vocoder models (MelGAN, Multiband-MelGAN, GAN-TTS, ParallelWaveGAN)
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

## Main Requirements and Installation
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python>=3.6
  * pytorch>=1.5.0
  * tensorflow>=2.3
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode

Install TTS using ```setup.py```. It will install all of the requirements automatically and make TTS available to all the python environment as an ordinary python module.

```python setup.py develop```

Or you can use ```requirements.txt``` to install the requirements only.

```pip install -r requirements.txt```

### Directory Structure
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

### Docker
A docker image is created by [@synesthesiam](https://github.com/synesthesiam) and shared in a separate [repository](https://github.com/synesthesiam/docker-mozillatts) with the latest LJSpeech models.

## Release Models
Please visit [our wiki.](https://github.com/mozilla/TTS/wiki/Released-Models)

## Sample Model Output
Below you see Tacotron model state after 16K iterations with batch-size 32 with LJSpeech dataset.

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio examples: [soundcloud](https://soundcloud.com/user-565970875/pocket-article-wavernn-and-tacotron2)

<img src="images/example_model_output.png?raw=true" alt="example_output" width="400"/>

## [Mozilla TTS Tutorials and Notebooks](https://github.com/mozilla/TTS/wiki/TTS-Notebooks-and-Tutorials)

## Datasets and Data-Loading
TTS provides a generic dataloader easy to use for your custom dataset.
You just need to write a simple function to format the dataset. Check ```datasets/preprocess.py``` to see some examples.
After that, you need to set ```dataset``` fields in ```config.json```.

Some of the public datasets that we successfully applied TTS:

- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- [Nancy](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
- [TWEB](https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset)
- [M-AI-Labs](http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
- [LibriTTS](https://openslr.org/60/)
- [Spanish](https://drive.google.com/file/d/1Sm_zyBo67XHkiFhcRSQ4YaHPYM0slO_e/view?usp=sharing) - thx! @carlfm01

## Training and Fine-tuning LJ-Speech
Here you can find a [CoLab](https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b) notebook for a hands-on example, training LJSpeech. Or you can manually follow the guideline below.

To start with, split ```metadata.csv``` into train and validation subsets respectively ```metadata_train.csv``` and ```metadata_val.csv```. Note that for text-to-speech, validation performance might be misleading since the loss value does not directly measure the voice quality to the human ear and it also does not measure the attention module performance. Therefore, running the model with new sentences and listening to the results is the best way to go.

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 1100 metadata_shuf.csv > metadata_val.csv
```

To train a new model, you need to define your own ```config.json``` to define model details, trainin configuration and more (check the examples). Then call the corressponding train script.

For instance, in order to train a tacotron or tacotron2 model on LJSpeech dataset, follow these steps.

```python TTS/bin/train_tacotron.py --config_path TTS/tts/configs/config.json```

To fine-tune a model, use ```--restore_path```.

```python TTS/bin/train_tacotron.py --config_path TTS/tts/configs/config.json --restore_path /path/to/your/model.pth.tar```

To continue an old training run, use ```--continue_path```.

```python TTS/bin/train_tacotron.py --continue_path /path/to/your/run_folder/```

For multi-GPU training, call ```distribute.py```. It runs any provided train script in multi-GPU setting.

```CUDA_VISIBLE_DEVICES="0,1,4" python TTS/bin/distribute.py --script train_tacotron.py --config_path TTS/tts/configs/config.json```

Each run creates a new output folder accomodating used ```config.json```, model checkpoints and tensorboard logs.

In case of any error or intercepted execution, if there is no checkpoint yet under the output folder, the whole folder is going to be removed.

You can also enjoy Tensorboard,  if you point Tensorboard argument```--logdir``` to the experiment folder.

## Contribution guidelines
This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the [Mozilla Community Participation Guidelines.](https://www.mozilla.org/about/governance/policies/participation/)

Please send your Pull Request to ```dev``` branch. Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```

## Collaborative Experimentation Guide
If you like to use TTS to try a new idea and like to share your experiments with the community, we urge you to use the following guideline for a better collaboration.
(If you have an idea for better collaboration, let us know)
- Create a new branch.
- Open an issue pointing your branch.
- Explain your experiment.
- Share your results as you proceed. (Tensorboard log files, audio results, visuals etc.)
- Use LJSpeech dataset (for English) if you like to compare results with the released models. (It is the most open scalable dataset for quick experimentation)

## [Contact/Getting Help](https://github.com/mozilla/TTS/wiki/Contact-and-Getting-Help)

## Major TODOs
- [x] Implement the model.
- [x] Generate human-like speech on LJSpeech dataset.
- [x] Generate human-like speech on a different dataset (Nancy) (TWEB).
- [x] Train TTS with r=1 successfully.
- [x] Enable process based distributed training. Similar to (https://github.com/fastai/imagenet-fast/).
- [x] Adapting Neural Vocoder. TTS works with WaveRNN and ParallelWaveGAN (https://github.com/erogol/WaveRNN and https://github.com/erogol/ParallelWaveGAN)
- [x] Multi-speaker embedding.
- [x] Model optimization (model export, model pruning etc.)

<!--## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
-->

### Acknowledgement
- https://github.com/keithito/tacotron (Dataset pre-processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
- https://github.com/kan-bayashi/ParallelWaveGAN (vocoder library)
- https://github.com/jaywalnut310/glow-tts (Original Glow-TTS implementation)
- https://github.com/fatchord/WaveRNN/ (Original WaveRNN implementation)
