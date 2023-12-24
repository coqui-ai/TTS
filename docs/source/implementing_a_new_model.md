# Implementing a Model

1. Implement layers.

    You can either implement the layers under `TTS/tts/layers/new_model.py` or in the model file `TTS/tts/model/new_model.py`.
    You can also reuse layers already implemented.

2. Test layers.

    We keep tests under `tests` folder. You can add `tts` layers tests under `tts_tests` folder.
    Basic tests are checking input-output tensor shapes and output values for a given input. Consider testing extreme cases that are more likely to cause problems like `zero` tensors.

3. Implement a loss function.

    We keep loss functions under `TTS/tts/layers/losses.py`. You can also mix-and-match implemented loss functions as you like.

   A loss function returns a dictionary in a format ```{’loss’: loss, ‘loss1’:loss1 ...}``` and the dictionary must at least define the `loss` key which is the actual value used by the optimizer. All the items in the dictionary are automatically logged on the terminal and the Tensorboard.

4. Test the loss function.

    As we do for the layers, you need to test the loss functions too. You need to check input/output tensor shapes,
    expected output values for a given input tensor. For instance, certain loss functions have upper and lower limits and
    it is a wise practice to test with the inputs that should produce these limits.

5. Implement `MyModel`.

    In 🐸TTS, a model class is a self-sufficient implementation of a model directing all the interactions with the other
    components. It is enough to implement the API provided by the `BaseModel` class to comply.

    A model interacts with the `Trainer API` for training, `Synthesizer API` for inference and testing.

    A 🐸TTS model must return a dictionary by the `forward()` and `inference()` functions. This dictionary must `model_outputs` key that is considered as the main model output by the `Trainer` and `Synthesizer`.

    You can place your `tts` model implementation under `TTS/tts/models/new_model.py` then inherit and implement the `BaseTTS`.

    There is also the `callback` interface by which you can manipulate both the model and the `Trainer` states. Callbacks give you
    an infinite flexibility to add custom behaviours for your model and training routines.

    For more details, see {ref}`BaseTTS <Base tts Model>` and :obj:`TTS.utils.callbacks`.

6. Optionally, define `MyModelArgs`.

    `MyModelArgs` is a 👨‍✈️Coqpit class that sets all the class arguments of the `MyModel`. `MyModelArgs` must have
    all the fields necessary to instantiate the `MyModel`. However, for training, you need to pass `MyModelConfig` to
    the model.

7. Test `MyModel`.

    As the layers and the loss functions, it is recommended to test your model. One smart way for testing is that you
    create two models with the exact same weights. Then we run a training loop with one of these models and
    compare the weights with the other model. All the weights need to be different in a passing test. Otherwise, it
    is likely that a part of the model is malfunctioning or not even attached to the model's computational graph.

8. Define `MyModelConfig`.

    Place `MyModelConfig` file under `TTS/models/configs`. It is enough to inherit the `BaseTTSConfig` to make your
    config compatible with the `Trainer`. You should also include `MyModelArgs` as a field if defined. The rest of the fields should define the model
    specific values and parameters.

9. Write Docstrings.

    We love you more when you document your code. ❤️


# Template 🐸TTS Model implementation

You can start implementing your model by copying the following base class.

```python
from TTS.tts.models.base_tts import BaseTTS


class MyModel(BaseTTS):
    """
    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    def __init__(self, config: Coqpit):
        super().__init__()
        self._set_model_args(config)

    def _set_model_args(self, config: Coqpit):
        """Set model arguments from the config. Override this."""
        pass

    def forward(self, input: torch.Tensor, *args, aux_input={}, **kwargs) -> Dict:
        """Forward pass for the model mainly used in training.

        You can be flexible here and use different number of arguments and argument names since it is intended to be
        used by `train_step()` without exposing it out of the model.

        Args:
            input (torch.Tensor): Input tensor.
            aux_input (Dict): Auxiliary model inputs like embeddings, durations or any other sorts of inputs.

        Returns:
            Dict: Model outputs. Main model output must be named as "model_outputs".
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def inference(self, input: torch.Tensor, aux_input={}) -> Dict:
        """Forward pass for inference.

        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def train_log(self, batch: Dict, outputs: Dict, logger: "Logger", assets:Dict, steps:int) -> None:
        """Create visualizations and waveform examples for training.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previous training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        pass

    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single evaluation step. Run the model forward pass and compute losses. In most cases, you can
        call `train_step()` with no changes.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def eval_log(self, batch: Dict, outputs: Dict, logger: "Logger", assets:Dict, steps:int) -> None:
        """The same as `train_log()`"""
        pass

    def load_checkpoint(self, config: Coqpit, checkpoint_path: str, eval: bool = False) -> None:
        """Load a checkpoint and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
        """
        ...

    def get_optimizer(self) -> Union["Optimizer", List["Optimizer"]]:
        """Setup a return optimizer or optimizers."""
        pass

    def get_lr(self) -> Union[float, List[float]]:
        """Return learning rate(s).

        Returns:
            Union[float, List[float]]: Model's initial learning rates.
        """
        pass

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        pass

    def get_criterion(self):
        pass

    def format_batch(self):
        pass

```
