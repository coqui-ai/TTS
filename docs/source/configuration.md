# Configuration

We use 👩‍✈️[Coqpit] for configuration management. It provides basic static type checking and serialization capabilities on top of native Python `dataclasses`. Here is how a simple configuration looks like with Coqpit.

```python
from dataclasses import asdict, dataclass, field
from typing import List, Union
from coqpit.coqpit import MISSING, Coqpit, check_argument


@dataclass
class SimpleConfig(Coqpit):
    val_a: int = 10
    val_b: int = None
    val_d: float = 10.21
    val_c: str = "Coqpit is great!"
    vol_e: bool = True
    # mandatory field
    # raise an error when accessing the value if it is not changed. It is a way to define
    val_k: int = MISSING
    # optional field
    val_dict: dict = field(default_factory=lambda: {"val_aa": 10, "val_ss": "This is in a dict."})
    # list of list
    val_listoflist: List[List] = field(default_factory=lambda: [[1, 2], [3, 4]])
    val_listofunion: List[List[Union[str, int, bool]]] = field(
        default_factory=lambda: [[1, 3], [1, "Hi!"], [True, False]]
    )

    def check_values(
        self,
    ):  # you can define explicit constraints manually or by`check_argument()`
        """Check config fields"""
        c = asdict(self)  # avoid unexpected changes on `self`
        check_argument("val_a", c, restricted=True, min_val=10, max_val=2056)
        check_argument("val_b", c, restricted=True, min_val=128, max_val=4058, allow_none=True)
        check_argument("val_c", c, restricted=True)
```

In TTS, each model must have a configuration class that exposes all the values necessary for its lifetime.

It defines model architecture, hyper-parameters, training, and inference settings. For our models, we merge all the fields in a single configuration class for ease. It may not look like a wise practice but enables easier bookkeeping and reproducible experiments.

The general configuration hierarchy looks like below:

```
ModelConfig()
     |
     | -> ...         # model specific configurations
     | -> ModelArgs()           # model class arguments
     | -> BaseDatasetConfig()   # only for tts models
     | -> BaseXModelConfig()    # Generic fields for `tts` and `vocoder` models.
                |
                | -> BaseTrainingConfig()   # trainer fields
                | -> BaseAudioConfig()      # audio processing fields
```

In the example above, ```ModelConfig()``` is the final configuration that the model receives and it has all the fields necessary for the model.

We host pre-defined model configurations under ```TTS/<model_class>/configs/```. Although we recommend a unified config class, you can decompose it as you like as for your custom models as long as all the fields for the trainer, model, and inference APIs are provided.
