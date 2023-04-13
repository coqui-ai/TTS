from coqpit import Coqpit

from TTS.model import BaseTrainerModel

# pylint: skip-file


class BaseVocoder(BaseTrainerModel):
    """Base `vocoder` class. Every new `vocoder` model must inherit this.

    It defines `vocoder` specific functions on top of `Model`.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    MODEL_TYPE = "vocoder"

    def __init__(self, config):
        super().__init__()
        self._set_model_args(config)

    def _set_model_args(self, config: Coqpit):
        """Setup model args based on the config type.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign the directly.
        """
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            if "characters" in config:
                _, self.config, num_chars = self.get_characters(config)
                self.config.num_chars = num_chars
                if hasattr(self.config, "model_args"):
                    config.model_args.num_chars = num_chars
                    if "model_args" in config:
                        self.args = self.config.model_args
                    # This is for backward compatibility
                    if "model_params" in config:
                        self.args = self.config.model_params
            else:
                self.config = config
                if "model_args" in config:
                    self.args = self.config.model_args
                # This is for backward compatibility
                if "model_params" in config:
                    self.args = self.config.model_params
        else:
            raise ValueError("config must be either a *Config or *Args")
