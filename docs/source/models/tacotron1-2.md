# 🌮 Tacotron 1 and 2

Tacotron is one of the first successful DL-based text-to-mel models and opened up the whole TTS field for more DL research.

Tacotron mainly is an encoder-decoder model with attention.

The encoder takes input tokens (characters or phonemes) and the decoder outputs mel-spectrogram* frames. Attention module in-between learns to align the input tokens with the output mel-spectrgorams.

Tacotron1 and 2 are both built on the same encoder-decoder architecture but they use different layers. Additionally, Tacotron1 uses a Postnet module to convert mel-spectrograms to linear spectrograms with a higher resolution before the vocoder.

Vanilla Tacotron models are slow at inference due to the auto-regressive* nature that prevents the model to process all the inputs in parallel. One trick is to use a higher “reduction rate” that helps the model to predict multiple frames at once. That is, reduction rate 2 reduces the number of decoder iterations by half.

Tacotron also uses a Prenet module with Dropout that projects the model’s previous output before feeding it to the decoder again. The paper and most of the implementations use the Dropout layer even in inference and they report the attention fails or the voice quality degrades otherwise. But the issue with that, you get a slightly different output speech every time you run the model.

Training the attention is notoriously problematic in Tacoron models. Especially, in inference, for some input sequences, the alignment fails and causes the model to produce unexpected results. There are many different methods proposed to improve the attention.

After hundreds of experiments,  @ 🐸TTS we suggest Double Decoder Consistency that leads to the most robust model performance.

If you have a limited VRAM, then you can try using the Guided Attention Loss or the Dynamic Convolutional Attention. You can also combine the two.


## Important resources & papers
- Tacotron: https://arxiv.org/abs/2006.06873
- Tacotron2: https://arxiv.org/abs/2008.03802
- Double Decoder Consistency: https://coqui.ai/blog/tts/solving-attention-problems-of-tts-models-with-double-decoder-consistency
- Guided Attention Loss: https://arxiv.org/abs/1710.08969
- Forward & Backward Decoder: https://arxiv.org/abs/1907.09006
- Forward Attention: https://arxiv.org/abs/1807.06736
- Gaussian Attention: https://arxiv.org/abs/1910.10288
- Dynamic Convolutional Attention: https://arxiv.org/pdf/1910.10288.pdf


## BaseTacotron
```{eval-rst}
.. autoclass:: TTS.tts.models.base_tacotron.BaseTacotron
    :members:
```

## Tacotron
```{eval-rst}
.. autoclass:: TTS.tts.models.tacotron.Tacotron
    :members:
```

## Tacotron2
```{eval-rst}
.. autoclass:: TTS.tts.models.tacotron2.Tacotron2
    :members:
```

## TacotronConfig
```{eval-rst}
.. autoclass:: TTS.tts.configs.tacotron_config.TacotronConfig
    :members:
```

## Tacotron2Config
```{eval-rst}
.. autoclass:: TTS.tts.configs.tacotron2_config.Tacotron2Config
    :members:
```


