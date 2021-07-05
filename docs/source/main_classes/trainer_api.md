# Trainer API

The {class}`TTS.trainer.Trainer` provides a lightweight, extensible, and feature-complete training run-time. We optimized it for 🐸 but
can also be used for any DL training in different domains. It supports distributed multi-gpu, mixed-precision (apex or torch.amp) training.


## Trainer
```{eval-rst}
.. autoclass:: TTS.trainer.Trainer
    :members:
```

## TrainingArgs
```{eval-rst}
.. autoclass:: TTS.trainer.TrainingArgs
    :members:
```