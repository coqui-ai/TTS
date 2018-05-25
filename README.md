# TTS (Work in Progress...)
This project is a part of [Mozilla Common Voice](https://voice.mozilla.org/en). TTS targets a Text2Speech engine lightweight in computation with high quality speech synthesis. You might hear a sample [here]().

Here we have pytorch implementation of Tacotron: [A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135). We plan to improve the model by the time with new architectural updates.

You can find [here](http://www.erogol.com/text-speech-deep-learning-architectures/) a brief note pointing possible TTS architectures and their comparisons.

## Requirements
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python 3.6
  * pytorch 0.4
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode
 
## Checkpoints and Audio Samples
Checkout [here](https://mycroft.ai/blog/available-voices/#the-human-voice-is-the-most-perfect-instrument-of-all-arvo-part) to compare the samples (except the first) below.

| Models        | Commit            | Audio Sample  |
| ------------- |:-----------------:|:-------------|
| [iter-62410](https://drive.google.com/open?id=1pjJNzENL3ZNps9n7k_ktGbpEl6YPIkcZ)| [99d56f7](https://github.com/mozilla/TTS/tree/99d56f7e93ccd7567beb0af8fcbd4d24c48e59e9)           | [link](https://soundcloud.com/user-565970875/99d56f7-iter62410 )|
| [iter-170K](https://drive.google.com/open?id=16L6JbPXj6MSlNUxEStNn28GiSzi4fu1j) | [e00bc66](https://github.com/mozilla/TTS/tree/e00bc66) |[link](https://soundcloud.com/user-565970875/april-13-2018-07-06pm-e00bc66-iter170k)|
| Best: [iter-270K](https://drive.google.com/drive/folders/1Q6BKeEkZyxSGsocK2p_mqgzLwlNvbHFJ?usp=sharing)|[256ed63](https://github.com/mozilla/TTS/tree/256ed63)|[link](https://soundcloud.com/user-565970875/sets/samples-1650226)|

## Data
Currently TTS provides data loaders for
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

## Training the network
To run your own training, you need to define a ```config.json``` file (simple template below) and call with the command.

```train.py --config_path config.json```

If you like to use specific set of GPUs.

```CUDA_VISIBLE_DEVICES="0,1,4" train.py --config_path config.json```

Each run creates an experiment folder with the date and the time, under the folder you set in ```config.json```. And if there is no checkpoint yet under that folder, it is going to be removed when you exit the training or an error is raised.

You can also enjoy Tensorboard with couple of good training indicators, if you point the Tensorboard argument```--logdir``` to the experiment folder.

Example ```config.json```:
```
{
  "model_name": "my-model", // used in the experiment folder name
  "num_mels": 80,
  "num_freq": 1025,
  "sample_rate": 20000,
  "frame_length_ms": 50,
  "frame_shift_ms": 12.5,
  "preemphasis": 0.97,
  "min_level_db": -100,
  "ref_level_db": 20,
  "embedding_size": 256,
  "text_cleaner": "english_cleaners",

  "epochs": 1000,
  "lr": 0.002,
  "warmup_steps": 4000,
  "batch_size": 32,
  "eval_batch_size":32,
  "r": 5,
    
  "griffin_lim_iters": 60,
  "power": 1.5,

  "num_loader_workers": 8,

  "checkpoint": true,
  "save_step": 376,
  "data_path": "/my/training/data/path",
  "min_seq_len": 0, 
  "output_path": "/my/experiment/folder/path"
}

```

## Testing
Best way to test your pretrained network is to use Notebooks under ```notebooks``` folder. 

## Contribution
Any kind of contribution is highly welcome as we are propelled by the open-source spirit. If you like to add or edit things in code, please also consider to write tests to verify your segment so that we can be sure things are on track as this repo gets bigger. 

## TODO
Checkout issues and Project field.

## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482) 
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
        
### Precursor implementations
- https://github.com/keithito/tacotron (Dataset and Test processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
