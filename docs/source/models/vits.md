# VITS

VITS (Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech
) is an End-to-End (encoder -> vocoder together) TTS model that takes advantage of SOTA DL techniques like GANs, VAE,
Normalizing Flows. It does not require external alignment annotations and learns the text-to-audio alignment
using MAS as explained in the paper. The model architecture is a combination of GlowTTS encoder and HiFiGAN vocoder.
It is a feed-forward model with x67.12 real-time factor on a GPU.

## Important resources & papers
- VITS: https://arxiv.org/pdf/2106.06103.pdf
- Neural Spline Flows: https://arxiv.org/abs/1906.04032
- Variational Autoencoder: https://arxiv.org/pdf/1312.6114.pdf
- Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
- HiFiGAN: https://arxiv.org/abs/2010.05646
- Normalizing Flows: https://blog.evjang.com/2018/01/nf1.html

## VitsConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.vits_config.VitsConfig
    :members:
```

## VitsArgs
```{eval-rst}
.. autoclass:: TTS.tts.models.vits.VitsArgs
    :members:
```

## Vits Model
```{eval-rst}
.. autoclass:: TTS.tts.models.vits.Vits
    :members:
```
