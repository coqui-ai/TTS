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

1. Define `train.py`.

    ```python
    import os

    # GlowTTSConfig: all model related values for training, validating and testing.
    from TTS.tts.configs.glow_tts_config import GlowTTSConfig

    # BaseDatasetConfig: defines name, formatter and path of the dataset.
    from TTS.tts.configs.shared_config import BaseDatasetConfig

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

    # initialize the audio processor used for feature extraction and audio I/O.
    # It is mainly used by the dataloader and the training loggers.
    ap = AudioProcessor(**config.audio.to_dict())

    # load a list of training samples
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

    # initialize the model
    # Models only takes the config object as input.
    model = GlowTTS(config)

    # Initiate the Trainer.
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainingArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
    )

    # And kick it 🚀
    trainer.fit()
    ```

2. Run the script.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python train.py
    ```

    - Continue a previous run.

        ```bash
        CUDA_VISIBLE_DEVICES=0 python train.py --continue_path path/to/previous/run/folder/
        ```

    - Fine-tune a model.

        ```bash
        CUDA_VISIBLE_DEVICES=0 python train.py --restore_path path/to/model/checkpoint.pth.tar
        ```

    - Run multi-gpu training.

        ```bash
        CUDA_VISIBLE_DEVICES=0,1,2 python TTS/bin/distribute.py --script train.py
        ```

### CLI Way

We still support running training from CLI like in the old days. The same training run can also be started as follows.

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

from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = os.path.dirname(os.path.abspath(__file__))

config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
```

❗️ Note that you can also use ```train_vocoder.py``` as the ```tts``` models above.

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



