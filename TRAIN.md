# Training Guide

In order to start training we first need to install coqui TTS. The main guide is [here](https://tts.readthedocs.io/en/latest/installation.html), however here is an alternative version which includes an explanation for preparing the training data.

## Installation

We will install coqui TTS from git in a virtualenv.

```
git clone https://github.com/CollectivaT-dev/TTS
cd TTS
virtualenv --python=python3 venv
```
coqui TTS works best with `python3.7`. After the installation we activate the virtualenv and 'nstall the dependencies.

```
source venv/bin/activate
make system-deps
pip install -e .[all]
```
`make system-deps` needs `sudo`. To test the installation do:

```
tts --model_name "tts_models/en/ljspeech/speedy-speech" --vocoder_name "vocoder_models/en/ljspeech/univnet" --text "This is a sample text for my model to speak."
```

## Fine tuning an English model (Arctic voice KSP)

For trial we will finetune the ljspeech speedy-speech model with the KSP Arctic voice (English with Indian accent). Before the training we will need to download the ljspeech speedy-speech model. If you have synthesized text with the last step of the installation, you will have already downloaded the model. Please note the where the model is downloaded, because we will need the path for the training. Usually it is downloaded to `home/<username>/.local/share/tts/tts_models--en--ljspeech--speedy-speech`.

Now we need to modify the hardcoded paths in the `config.json` file prepared for the arctic dataset. 

```
sed 's|/content/TTS|'`pwd`'|g' recipes/arctic/speedy_speech/config.json > recipes/arctic/speedy_speech/config_local.json
```
We will use this `config_local.json` file for the training.

Finally we will download the data and change the name of its directory. From `TTS` top directory:

```
wget http://laklak.eu/share/adaptation_data.tar.gz
tar xzf adaptation_data.tar.gz -C recipes/arctic/arctic-data/
mv recipes/arctic/arctic-data/adaptation_wav/ recipes/arctic/arctic-data/wavs
```

Finally we are ready to finetune the ljspeech model. We just need to launch the `TTS/bin/train_tts.py` with the relevant paths:

```
CUDA_VISIBLE_DEVICES="0"  python TTS/bin/train_tts.py --config_path  recipes/arctic/speedy_speech/config_local.json --restore_path  /home/<username>/.local/share/tts/tts_models--en--ljspeech--speedy-speech/model_file.pth.tar
```

The hyperparameters can be modified directly in the `config_local.json` or can be overwritten as an argument in `train_tts.py`. For example to change the learning rate and save step for the checkpoints (the current learning rate is already lowered for finetuning):

```
CUDA_VISIBLE_DEVICES="0"  python TTS/bin/train_tts.py \
	--config_path  recipes/arctic/speedy_speech/config_local.json \
	--restore_path  /home/<username>/.local/share/tts/tts_models--en--ljspeech--speedy-speech/model_file.pth.tar \
	--coqpit.lr 0.00001 \
	--coqpit.save_step 10000
```

This is for a single GPU training, there is also a possibility to train on multiple GPUs via `TTS/bin/distribute.py`. For that, please consult the original documentation [here](https://tts.readthedocs.io/en/latest/training_a_model.html).
