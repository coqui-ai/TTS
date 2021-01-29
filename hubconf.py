dependencies = ['torch', 'gdown']
import torch
import os
import zipfile

from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager



def tts(model_name='tts_models/en/ljspeech/tacotron2-DCA', vocoder_name='vocoder_models/en/ljspeech/mulitband-melgan', pretrained=True):
    manager = ModelManager()
  
    model_path, config_path = manager.download_model(model_name)
    vocoder_path, vocoder_config_path = manager.download_model(vocoder_name)
    
    # create synthesizer
    synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path)
    return synthesizer


if __name__ == '__main__':
    # synthesizer = torch.hub.load('/data/rw/home/projects/TTS/TTS', 'tts', source='local')
    synthesizer = torch.hub.load('mozilla/TTS:hub_conf', 'tts', source='github')
    synthesizer.tts("This is a test!")