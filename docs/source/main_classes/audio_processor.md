# AudioProcessor API

`TTS.utils.audio.AudioProcessor` is the core class for all the audio processing routines. It provides an API for

- Feature extraction.
- Sound normalization.
- Reading and writing audio files.
- Sampling audio signals.
- Normalizing and denormalizing audio signals.
- Griffin-Lim vocoder.

The `AudioProcessor` needs to be initialized with `TTS.config.shared_configs.BaseAudioConfig`. Any model config
also must inherit or initiate `BaseAudioConfig`.

## AudioProcessor
```{eval-rst}
.. autoclass:: TTS.utils.audio.AudioProcessor
    :members:
```

## BaseAudioConfig
```{eval-rst}
.. autoclass:: TTS.config.shared_configs.BaseAudioConfig
    :members:
```