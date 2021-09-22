# Glow TTS

Glow TTS is a normalizing flow model for text-to-speech. It is built on the generic Glow model that is previously
used in computer vision and vocoder models. It uses "monotonic alignment search" (MAS) to fine the text-to-speech alignment
and uses the output to train a separate duration predictor network for faster inference run-time.

## Important resources & papers
- GlowTTS: https://arxiv.org/abs/2005.11129
- Glow (Generative Flow with invertible 1x1 Convolutions): https://arxiv.org/abs/1807.03039
- Normalizing Flows: https://blog.evjang.com/2018/01/nf1.html

## GlowTTS Config
```{eval-rst}
.. autoclass:: TTS.tts.configs.glow_tts_config.GlowTTSConfig
    :members:
```

## GlowTTS Model
```{eval-rst}
.. autoclass:: TTS.tts.models.glow_tts.GlowTTS
    :members:
```
