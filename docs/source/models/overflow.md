# Overflow TTS

Neural HMMs are a type of neural transducer recently proposed for
sequence-to-sequence modelling in text-to-speech. They combine the best features
of classic statistical speech synthesis and modern neural TTS, requiring less
data and fewer training updates, and are less prone to gibberish output caused
by neural attention failures. In this paper, we combine neural HMM TTS with
normalising flows for describing the highly non-Gaussian distribution of speech
acoustics. The result is a powerful, fully probabilistic model of durations and
acoustics that can be trained using exact maximum likelihood. Compared to
dominant flow-based acoustic models, our approach integrates autoregression for
improved modelling of long-range dependences such as utterance-level prosody.
Experiments show that a system based on our proposal gives more accurate
pronunciations and better subjective speech quality than comparable methods,
whilst retaining the original advantages of neural HMMs. Audio examples and code
are available at https://shivammehta25.github.io/OverFlow/.


## Important resources & papers
- HMM: https://de.wikipedia.org/wiki/Hidden_Markov_Model
- OverflowTTS paper: https://arxiv.org/abs/2211.06892
- Neural HMM: https://arxiv.org/abs/2108.13320
- Audio Samples: https://shivammehta25.github.io/OverFlow/


## OverflowConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.overflow_config.OverflowConfig
    :members:
```

## Overflow Model
```{eval-rst}
.. autoclass:: TTS.tts.models.overflow.Overflow
    :members:
```