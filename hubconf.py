dependencies = ['torch', 'gdown', 'pysbd', 'phonemizer', 'unidecode']  # apt install espeak
import torch

from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager


def tts(model_name='tts_models/en/ljspeech/tacotron2-DCA', vocoder_name='vocoder_models/en/ljspeech/mulitband-melgan', use_cuda=False):
    """TTS entry point for PyTorch Hub that provides a Synthesizer object to synthesize speech from a give text.

    Example:
        >>> synthesizer = torch.hub.load('mozilla/TTS', 'tts', source='github')
        >>> wavs = synthesizer.tts("This is a test! This is also a test!!")
            wavs - is a list of values of the synthesized speech.

    Args:
        model_name (str, optional): One of the model names from .model.json. Defaults to 'tts_models/en/ljspeech/tacotron2-DCA'.
        vocoder_name (str, optional): One of the model names from .model.json. Defaults to 'vocoder_models/en/ljspeech/mulitband-melgan'.
        pretrained (bool, optional): [description]. Defaults to True.

    Returns:
        TTS.utils.synthesizer.Synthesizer: Synthesizer object wrapping both vocoder and tts models.
    """
    manager = ModelManager()

    model_path, config_path = manager.download_model(model_name)
    vocoder_path, vocoder_config_path = manager.download_model(vocoder_name)

    # create synthesizer
    synt = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, use_cuda)
    return synt


if __name__ == '__main__':
    synthesizer = torch.hub.load('mozilla/TTS:hub_conf', 'tts', source='github')
    synthesizer.tts("This is a test!")
