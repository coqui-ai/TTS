# Forward TTS model(s)

A general feed-forward TTS model implementation that can be configured to different architectures by setting different
encoder and decoder networks. It can be trained with either pre-computed durations (from pre-trained Tacotron) or
an alignment network that learns the text to audio alignment from the input data.

Currently we provide the following pre-configured architectures:

- **FastSpeech:**

    It's a feed-forward model TTS model that uses Feed Forward Transformer (FFT) modules as the encoder and decoder.

- **FastPitch:**

    It uses the same FastSpeech architecture that is conditioned on fundamental frequency (f0) contours with the
    promise of more expressive speech.

- **SpeedySpeech:**

    It uses Residual Convolution layers instead of Transformers that leads to a more compute friendly model.

- **FastSpeech2 (TODO):**

    Similar to FastPitch but it also uses a spectral energy values as an addition.

## Important resources & papers
- FastPitch: https://arxiv.org/abs/2006.06873
- SpeedySpeech: https://arxiv.org/abs/2008.03802
- FastSpeech: https://arxiv.org/pdf/1905.09263
- FastSpeech2: https://arxiv.org/abs/2006.04558
- Aligner Network: https://arxiv.org/abs/2108.10447
- What is Pitch: https://www.britannica.com/topic/pitch-speech


## ForwardTTSArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.forward_tts.ForwardTTSArgs
    :members:
```

## ForwardTTS Model
```{eval-rst}
.. autoclass:: TTS.tts.models.forward_tts.ForwardTTS
    :members:
```

## FastPitchConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.fast_pitch_config.FastPitchConfig
    :members:
```

## SpeedySpeechConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.speedy_speech_config.SpeedySpeechConfig
    :members:
```

## FastSpeechConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.fast_speech_config.FastSpeechConfig
    :members:
```


