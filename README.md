# Tacotron-pytorch

A pytorch implementation of [Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135).

<img src="png/model.png">

## Requirements
  * Install python 3
  * Install pytorch == 0.2.0
  * Install requirements:
    ```
   	pip install -r requirements.txt
   	```

## Data
I used LJSpeech dataset which consists of pairs of text script and wav files. The complete dataset (13,100 pairs) can be downloaded [here](https://keithito.com/LJ-Speech-Dataset/). I referred https://github.com/keithito/tacotron for the preprocessing code.

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `data.py` loads training data and preprocess text to index and wav files to spectrogram. Preprocessing codes for text is in text/ directory.
  * `module.py` contains all methods, including CBHG, highway, prenet, and so on.
  * `network.py` contains networks including encoder, decoder and post-processing network.
  * `train.py` is for training.
  * `synthesis.py` is for generating TTS sample.

## Training the network
  * STEP 1. Download and extract LJSpeech data at any directory you want.
  * STEP 2. Adjust hyperparameters in `hyperparams.py`, especially 'data_path' which is a directory that you extract files, and the others if necessary.
  * STEP 3. Run `train.py`. 

## Generate TTS wav file
  * STEP 1. Run `synthesis.py`. Make sure the restore step. 

## Samples
  * You can check the generated samples in 'samples/' directory. Training step was only 60K, so the performance is not good yet.

## Reference
  * Keith ito: https://github.com/keithito/tacotron

## Comments
  * Any comments for the codes are always welcome.

