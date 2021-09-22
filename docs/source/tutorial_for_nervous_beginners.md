# Tutorial For Nervous Beginners

## Installation

User friendly installation. Recommended only for synthesizing voice.

```bash
$ pip install TTS
```

Developer friendly installation.

```bash
$ git clone https://github.com/coqui-ai/TTS
$ cd TTS
$ pip install -e .
```

## Training a `tts` Model

A breakdown of a simple script training a GlowTTS model on LJspeech dataset. See the comments for the explanation of
each line.

### Pure Python Way

```python
import os

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs import BaseDatasetConfig

# init_training: Initialize and setup the training environment.
# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from TTS.trainer import init_training, Trainer, TrainingArgs

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))

# set LJSpeech as our target dataset and define its path so that the Trainer knows what data formatter it needs.
dataset_config = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSpeech-1.1/"))

# Configure the model. Every config class inherits the BaseTTSConfig to have all the fields defined for the Trainer.
config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config]
)

# Take the config and the default Trainer arguments, setup the training environment and override the existing
# config values from the terminal. So you can do the following.
# >>> python train.py --coqpit.batch_size 128
args, config, output_path, _, _, _= init_training(TrainingArgs(), config)

# Initiate the Trainer.
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training etc.
trainer = Trainer(args, config, output_path)

# And kick it 🚀
trainer.fit()
```

### CLI Way

We still support running training from CLI like in the old days. The same training can be started as follows.

1. Define your `config.json`

    ```json
    {
        "run_name": "my_run",
        "model": "glow_tts",
        "batch_size": 32,
        "eval_batch_size": 16,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 4,
        "run_eval": true,
        "test_delay_epochs": -1,
        "epochs": 1000,
        "text_cleaner": "english_cleaners",
        "use_phonemes": false,
        "phoneme_language": "en-us",
        "phoneme_cache_path": "phoneme_cache",
        "print_step": 25,
        "print_eval": true,
        "mixed_precision": false,
        "output_path": "recipes/ljspeech/glow_tts/",
        "datasets":[{"name": "ljspeech", "meta_file_train":"metadata.csv", "path": "recipes/ljspeech/LJSpeech-1.1/"}]
    }
    ```

2. Start training.
    ```bash
    $ CUDA_VISIBLE_DEVICES="0" python TTS/bin/train_tts.py --config_path config.json
    ```



## Training a `vocoder` Model

```python
import os

from TTS.vocoder.configs import HifiganConfig
from TTS.trainer import init_training, Trainer, TrainingArgs


output_path = os.path.dirname(os.path.abspath(__file__))
config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    # `vocoder` only needs a data path and they read recursively all the `.wav` files underneath.
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)
args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, tb_logger)
trainer.fit()
```

❗️ Note that you can also start the training run from CLI as the `tts` model above.

## Synthesizing Speech

You can run `tts` and synthesize speech directly on the terminal.

```bash
$ tts -h # see the help
$ tts --list_models  # list the available models.
```

![cli.gif](https://github.com/coqui-ai/TTS/raw/main/images/tts_cli.gif)


You can call `tts-server` to start a local demo server that you can open it on
your favorite web browser and 🗣️.

```bash
$ tts-server -h # see the help
$ tts-server --list_models  # list the available models.
```
![server.gif](https://github.com/coqui-ai/TTS/raw/main/images/demo_server.gif)



