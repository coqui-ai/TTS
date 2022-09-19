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

A breakdown of a simple script that trains a GlowTTS model on the LJspeech dataset. See the comments for more details.

### Pure Python Way

0. Download your dataset.

    In this example, we download and use the LJSpeech dataset. Set the download directory based on your preferences.

    ```bash
    $ python -c 'from TTS.utils.downloaders import download_ljspeech; download_ljspeech("../recipes/ljspeech/");'
    ```

1. Define `train.py`.

    ```{literalinclude} ../../recipes/ljspeech/glow_tts/train_glowtts.py
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
        CUDA_VISIBLE_DEVICES=0 python train.py --restore_path path/to/model/checkpoint.pth
        ```

    - Run multi-gpu training.

        ```bash
        CUDA_VISIBLE_DEVICES=0,1,2 python -m trainer.distribute --script train.py
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
        "datasets":[{"formatter": "ljspeech", "meta_file_train":"metadata.csv", "path": "recipes/ljspeech/LJSpeech-1.1/"}]
    }
    ```

2. Start training.
    ```bash
    $ CUDA_VISIBLE_DEVICES="0" python TTS/bin/train_tts.py --config_path config.json
    ```

## Training a `vocoder` Model

```{literalinclude} ../../recipes/ljspeech/hifigan/train_hifigan.py
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
