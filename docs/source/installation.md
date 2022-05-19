# Installation

🐸TTS supports python >=3.7 <3.11.0 and tested on Ubuntu 18.10, 19.10, 20.10.

## Using `pip`

`pip` is recommended if you want to use 🐸TTS only for inference.

You can install from PyPI as follows:

```bash
pip install TTS  # from PyPI
```

Or install from Github:

```bash
pip install git+https://github.com/coqui-ai/TTS  # from Github
```

## Installing From Source

This is recommended for development and more control over 🐸TTS.

```bash
git clone https://github.com/coqui-ai/TTS/
cd TTS
make system-deps  # only on Linux systems.
make install
```

## On Windows
If you are on Windows, 👑@GuyPaddock wrote installation instructions [here](https://stackoverflow.com/questions/66726331/