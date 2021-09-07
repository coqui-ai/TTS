# FastPitch

FastPitch is a feed-forward encoder-decoder TTS model. It computes mel-spectrogram from the given input character sequence.
It uses a duration predictor network to predict the duration of each input character in the output sequence. In the original paper, they use a pre-trained Tacotron model to generate the labels for the duration predictor. In this implementation, you can also use an aligner network to learn the durations from the data and train the duration predictor in parallel. Original FastPitch model uses FeedForwardTransformer networks for both encoder and decoder. But in this implementation, you have the freedom to choose different encoder and decoder networks by just changing the relevant fields in the model configuration. Please see `FastPitchArgs` and `FastPitchConfig` below for more details.

## Important resources & papers
- FastPitch: https://arxiv.org/abs/2006.06873
- FastSpeech: https://arxiv.org/pdf/1905.09263
- Aligner Network: https://arxiv.org/abs/2108.10447
- What is Pitch: https://www.britannica.com/topic/pitch-speech

## FastPitchConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.fast_pitch_config.FastPitchConfig
    :members:
```

## FastPitchArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.fast_pitch.FastPitchArgs
    :members:
```

## FastPitch Model
```{eval-rst}
.. autoclass:: TTS.tts.models.fast_pitch.FastPitch
    :members:
```
