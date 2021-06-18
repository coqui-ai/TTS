from TTS.model import BaseModel

# pylint: skip-file


class BaseVocoder(BaseModel):
    """Base `vocoder` class. Every new `vocoder` model must inherit this.

    It defines `vocoder` specific functions on top of `Model`.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    def __init__(self):
        super().__init__()
