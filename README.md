# TTS (Work in Progress...)
TTS targets a Text2Speech engine lightweight in computation with hight quality speech construction. 

Here we have pytorch implementation of Tacotron: [A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135) as the start point. We plan to improve the model by the recent updated at the field.  

You can find [here](https://www.evernote.com/shard/s146/sh/9544e7e9-d372-4610-a7b7-3ddcb63d5dac/d01d33837dab625229dec3cfb4cfb887) a brief note pointing possible TTS architectures and their comparisons.

## Requirements
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python 3.6
  * pytorch > 0.2.0
  * TODO

## Audio Samples
All samples below are generated in test setting and can be generated a model trained on the master branch.
  * https://soundcloud.com/user-565970875/tts-ljspeech-val-35000 (iteration 35000)
  * https://soundcloud.com/user-565970875/tts-ljspeech-val-13585
  
## Checkpoints

| Models        | Commit            | Audio Sample  |
| ------------- |:-----------------:|:-------------|
| [iter-62410](https://drive.google.com/open?id=1pjJNzENL3ZNps9n7k_ktGbpEl6YPIkcZ)      | [99d56f7](https://github.com/mozilla/TTS/tree/99d56f7e93ccd7567beb0af8fcbd4d24c48e59e9)           | https://soundcloud.com/user-565970875/99d56f7-iter62410 |


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
  "num_mels": 80,
  "num_freq": 1025,
  "sample_rate": 22050,
  "frame_length_ms": 50,
  "frame_shift_ms": 12.5,
  "preemphasis": 0.97,
  "min_level_db": -100,
  "ref_level_db": 20,
  "embedding_size": 256,
  "text_cleaner": "english_cleaners",

  "epochs": 200,
  "lr": 0.002,
  "warmup_steps": 4000,
  "batch_size": 32,
  "eval_batch_size":32,
  "r": 5,

  "griffin_lim_iters": 60,
  "power": 1.2,

  "dataset": "TWEB",
  "meta_file_train": "transcript_train.txt",
  "meta_file_val": "transcript_val.txt",
  "data_path": "/data/shared/BibleSpeech/",
  "min_seq_len": 0, 
  "num_loader_workers": 8,

  "checkpoint": true,  // if save checkpoint per save_step
  "save_step": 200,
  "output_path": "/path/to/my_experiment",
}
```

## Testing
Best way to test your pretrained network is to use the Notebook under ```notebooks``` folder. 

## Contribution
Any kind of contribution is highly welcome as we are propelled by the open-source spirit. If you like to add or edit things in code, please also consider to write tests to verify your segment so that we can be sure things are on the track as this repo gets bigger. 

## TODO
- [x] Make the default Tacotron architecture functional with reasonable fidelity.
- [ ] Update the architecture with the latest improvements at the field. (e.g. Monotonic Attention, World Vocoder)
    - Using harmonized teacher forcing proposed by 
    - Update the attention module with a monotonic alternative. (e.g GMM attention, Window based attention)
    - References:
        - [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
        - [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
        - [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
        - [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
        - [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [ ] Simplify the architecture and push the limits of performance vs efficiency.
- [ ] Improve vocoder part of the network.
    - Possible Solutions:
        - WORLD vocoder
        - [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
        - [Faster WaveNet](https://arxiv.org/abs/1611.09482) 
        - [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
        
### Precursor implementations
- https://github.com/keithito/tacotron (Dataset and Test processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
