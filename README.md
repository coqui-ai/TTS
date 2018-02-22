# TTS (Work in Progress...)

Here we have pytorch implementation of: 
- Tacotron: [A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135).
- Tacotron2 (TODO): [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)

At the end, it should be easy to add new models and try different architectures.

You can find [here](https://www.evernote.com/shard/s146/sh/9544e7e9-d372-4610-a7b7-3ddcb63d5dac/d01d33837dab625229dec3cfb4cfb887) a brief note about possible TTS architectures and their comparisons. 

## Requirements
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python 3.6
  * pytorch > 0.2.0
  * TODO

## Data
Currently TTS provides data loaders for
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

## Training the network
To run your own training, you need to define a ```config.json``` file (simple template below) and call with the command.

```train.py --config_path config.json```

If you like to use specific set of GPUs.

```CUDA_VISIBLE_DEVICES="0,1,4" train.py --config_path config.json```

Each run creates an experiment folder with the corresponfing date and time, under the folder you set in ```config.json```. And if there is no checkpoint yet under that folder, it is going to be removed when you press Ctrl+C.

You can also enjoy Tensorboard with couple of good training logs, if you point ```--logdir``` the experiment folder.

Example ```config.json```:
```
{
  // Data loading parameters
  "num_mels": 80,
  "num_freq": 1024,
  "sample_rate": 20000,
  "frame_length_ms": 50.0,
  "frame_shift_ms": 12.5,
  "preemphasis": 0.97,
  "min_level_db": -100,
  "ref_level_db": 20,
  "hidden_size": 128,
  "embedding_size": 256,
  "text_cleaner": "english_cleaners",

  // Training parameters
  "epochs": 2000,
  "lr": 0.001,
  "batch_size": 256,
  "griffinf_lim_iters": 60,
  "power": 1.5,
  "r": 5,            // number of decoder outputs for Tacotron

  // Number of data loader processes
  "num_loader_workers": 8,

  // Experiment logging parameters
  "checkpoint": true,  // if save checkpoint per save_step
  "save_step": 200,
  "data_path": "/path/to/KeithIto/LJSpeech-1.0",
  "output_path": "/path/to/my_experiment",
  "log_dir": "/path/to/my/tensorboard/logs/"
}
```

## Testing
Best way to test your pretrained network is to use the Notebook under ```notebooks``` folder. 

## Contribution
Any kind of contribution is highly welcome as we are propelled by the open-source spirit. If you like to add or edit things in code, please also consider to write tests to verify your segment so that we can be sure things are on the track as this repo gets bigger. 
