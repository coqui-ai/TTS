# Implementing a Model

1. Implement layers.

    You can either implement the layers under `TTS/tts/layers/new_model.py` or in the model file `TTS/tts/model/new_model.py`.
    You can also reuse layers already implemented.

2. Test layers.

    We keep tests under `tests` folder. You can add `tts` layers tests under `tts_tests` folder.
    Basic tests are checking input-output tensor shapes and output values for a given input. Consider testing extreme cases that are more likely to cause problems like `zero` tensors.

3. Implement loss function.

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

    A 🐸TTS model must return a dictionary by the `forward()` and `inference()` functions. This dictionary must also include the `model_outputs` key that is considered as the main model output by the `Trainer` and `Synthesizer`.

    You can place your `tts` model implementation under `TTS/tts/models/new_model.py` then inherit and implement the `BaseTTS`.

    There is also the `callback` interface by which you can manipulate both the model and the `Trainer` states. Callbacks give you
    the infinite flexibility to add custom behaviours for your model and training routines.

    For more details, see {ref}`BaseTTS <Base TTS Model>` and :obj:`TTS.utils.callbacks`.

6. Optionally, define `MyModelArgs`.

    `MyModelArgs` is a 👨‍✈️Coqpit class that sets all the class arguments of the `MyModel`. It should be enough to pass
    an `MyModelArgs` instance to initiate the `MyModel`.

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
