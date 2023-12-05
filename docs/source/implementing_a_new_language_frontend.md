# Implementing a New Language Frontend

- Language frontends are located under `TTS.tts.utils.text`
- Each special language has a separate folder.
- Each folder contains all the utilities for processing the text input.
- `TTS.tts.utils.text.phonemizers` contains the main phonemizer for a language. This is the class that uses the utilities
from the previous step and used to convert the text to phonemes or graphemes for the model.
- After you implement your phonemizer, you need to add it to the `TTS/tts/utils/text/phonemizers/__init__.py` to be able to
map the language code in the model config - `config.phoneme_language` - to the phonemizer class and initiate the phonemizer automatically.
- You should also add tests to `tests/text_tests` if you want to make a PR.

We suggest you to check the available implementations as reference. Good luck!
