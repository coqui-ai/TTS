#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

import inquirer
from inquirer.themes import GreenPassion

#TODO add inquirer in requirements.txt

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

str2none = lambda i : i or None #converter for default None (''->None)

def official_zoo_inquirer():
    model_list={}
    model_list['tts_models']=manager.list_tts_models(print_list=False)
    model_list['vocoder_models']=manager.list_vocoder_models(print_list=False)
    model_list['vocoder_models'].insert(0,'default_vocoder')
    model_load_questions = [
    inquirer.List('use_cuda',
                message="Run model on CUDA?",
                choices=[True,False],
                default=False,
            ),
    inquirer.List('tts_choose',
                message="Choose a tts model to load",
                choices=model_list['tts_models'],
                default="tts_models/en/ljspeech/tacotron2-DDC"
            ),
    inquirer.List('vocoder_choose',
                message="Choose a vocoder model to load",
                choices=model_list['vocoder_models'],
            )
    ]
    answers_model_load = inquirer.prompt(model_load_questions, theme=GreenPassion())
    return answers_model_load

def official_zoo_info_inquirer():
    joint_model_list=manager.list_tts_models(print_list=False)+manager.list_vocoder_models(print_list=False)
    model_info_questions = [
    inquirer.List('model_choose_for_info',
                message="Choose a tts model for info",
                choices=joint_model_list,
            ),
    ]
    answers_model_info_request = inquirer.prompt(model_info_questions, theme=GreenPassion())
    return answers_model_info_request

def custom_model_inquirer():
    custom_model_load_questions = [
    inquirer.List('use_cuda',
                message="Run model on CUDA?",
                choices=[True,False],
                default=False,
            ),
    inquirer.Text('model_path',
                message="Path to TTS model path",
                default=None,
            ),
    inquirer.Text('model_config_path',
                message="Path to TTS model config path",
                default=None,
            ),
    inquirer.Text('vocoder_path',
                message="Path to vocoder model file.",
                default=None,
            ),
    inquirer.Text('vocoder_config_path',
                message="Path to vocoder model config file.",
                default=None,
            ),
    inquirer.Text('encoder_path',
                message="Path to speaker encoder model file.",
                default=None,
            ),
    inquirer.Text('encoder_config_path',
                message="Path to speaker encoder config file.",
                default=None,
            ),    
    ]
    answers_custom_model_load = inquirer.prompt(custom_model_load_questions, theme=GreenPassion())
    return answers_custom_model_load

def multispeaker_inquirer():
    multispeaker_questions = [
    inquirer.Text('speakers_file_path',
                message="JSON file for multi-speaker model.",
                default=None,
            ),
    inquirer.Text('language_ids_file_path',
                message="JSON file for multi-lingual model.",
                default=None,
            ),
    inquirer.Text('speaker_idx',
                message="Enter speaker idx",
                default=None
            ),
    inquirer.Text('language_idx',
                message="Enter language idx",
                default=None
            ),
    inquirer.Text('speaker_wav',
                message="Enter speaker wav file path",
                default=None
            ),
    inquirer.Text('reference_wav',
                message="Enter ref wav file path",
                default=None
            ),
    inquirer.Text('reference_speaker_idx',
                message="Enter reference speaker idx",
                default=None
            ),
    ]
    answers_multispeaker = inquirer.prompt(multispeaker_questions, theme=GreenPassion())
    return answers_multispeaker

def capacitron_inquirer():
    capacitron_questions = [
    inquirer.Text('capacitron_style_wav',
                message="Enter capacitron style wav path",
                default=None
            ),
    inquirer.Text('capacitron_style_text',
                message="Enter capacitron style text",
                default=None
            ),
    ]
    answers_capacitron = inquirer.prompt(capacitron_questions, theme=GreenPassion())
    return answers_capacitron

def continue_inquirer():
    continue_questions = [
        inquirer.List('to_do',
                        message="What to do next?",
                        choices=[
                            'try another text-input',
                            'restart tts',
                            'exit tts'
                        ]
                    ),
        ]

    continue_answers = inquirer.prompt(continue_questions, theme=GreenPassion())
    return continue_answers

def tts_inquirer(
    synthesizer,
    speaker_idx,
    language_idx,
    speaker_wav,
    reference_wav,
    reference_speaker_idx,
    capacitron_style_wav,
    capacitron_style_text
):
    tts_questions = [
    inquirer.Text('text_input',
                message="Type text to convert to speech",
            ),
    inquirer.Text('out_path',
                message="Enter output wav path",
                default="tts_output.wav"
            ),
    ]
    answers_tts = inquirer.prompt(tts_questions, theme=GreenPassion())
    text = str2none(answers_tts['text_input'])
    outpath = answers_tts['out_path']
    text = text if text is not None else "Enter random text."
    print(f" > Text: {text}")
    # kick it
    wav = synthesizer.tts(
        text,
        speaker_idx,
        language_idx,
        speaker_wav,
        reference_wav=reference_wav,
        reference_speaker_name=reference_speaker_idx,
        style_wav=capacitron_style_wav,
        style_text=capacitron_style_text,
    )

    # save the results
    print(f" > Saving output to {outpath}")
    synthesizer.save_wav(wav, outpath)

    continue_answers=continue_inquirer()
    return continue_answers

def block_prompt(synthesizer, tts_model_name):
    speaker_idx=None
    language_idx=None
    speaker_wav=None
    reference_wav=reference_wav=None
    reference_speaker_idx=None
    capacitron_style_wav=None
    capacitron_style_text=None

    if synthesizer.tts_speakers_file or hasattr(synthesizer.tts_model.speaker_manager, "ids"):
        answers_multispeaker=multispeaker_inquirer()
        for key,item in answers_multispeaker.items():
            answers_multispeaker[key]=str2none(item)
        speaker_idx=answers_multispeaker['speaker_idx']
        language_idx=answers_multispeaker['language_idx']
        speaker_wav=answers_multispeaker['speaker_wav']
        reference_wav=reference_wav=answers_multispeaker['reference_wav']
        reference_speaker_idx=answers_multispeaker['reference_speaker_idx']

    if 'capacitron' in tts_model_name:
        answers_capacitron = capacitron_inquirer()
        for key,item in answers_capacitron.items():
            answers_capacitron[key]=str2none(item)
        capacitron_style_wav=answers_capacitron['capacitron_style_wav']
        capacitron_style_text=answers_capacitron['capacitron_style_text']
    
    continue_answers = tts_inquirer(
            synthesizer,
            speaker_idx,
            language_idx,
            speaker_wav,
            reference_wav,
            reference_speaker_idx,
            capacitron_style_wav,
            capacitron_style_text
        )
    if continue_answers['to_do'] == 'exit tts':
        return
    if continue_answers['to_do'] == 'restart tts':
        print("restart")
        init_prompt()
    if continue_answers['to_do'] == 'try another text-input':
        block_prompt(synthesizer, tts_model_name)

def init_prompt():
    model_path=None
    config_path=None
    speakers_file_path=None
    language_ids_file_path=None
    vocoder_path=None
    vocoder_config_path=None
    encoder_path=None
    encoder_config_path=None
    use_cuda=False
          
    init_questions = [
    inquirer.List('to_do',
                    message="What do you need?",
                    choices=[ 
                        'get info from official model zoo',
                        'play with official model zoo',
                        'play with your own model',
                        'exit tts'
                    ]
                ),
    ]

    init_answers = inquirer.prompt(init_questions, theme=GreenPassion())

    if init_answers['to_do'] == 'get info from official model zoo':
        answers_model_load = official_zoo_info_inquirer()
        manager.model_info_by_full_name(answers_model_load['model_choose_for_info'])

    if init_answers['to_do'] == 'exit tts':
        return

    if init_answers['to_do'] == 'play with official model zoo':
        answers_model_load = official_zoo_inquirer()
        tts_model_name=answers_model_load['tts_choose']
        model_path, config_path, model_item = manager.download_model(tts_model_name)
        vocoder_name=answers_model_load['vocoder_choose'] if answers_model_load['vocoder_choose'] != "default_vocoder" else model_item["default_vocoder"]  
        if vocoder_name is not None:
            vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
        use_cuda=answers_model_load['use_cuda']

        synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        language_ids_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda,
        )

        block_prompt(synthesizer, tts_model_name)

    if init_answers['to_do'] == 'play with your own model':
        answers_custom_model_load = custom_model_inquirer()
        for key,item in answers_custom_model_load.items():
            answers_custom_model_load[key]=str2none(item)
        print(answers_custom_model_load)

def main():
    from TTS.bin.frogie import ascii_art_printer
    # ascii_art_printer()
    print("welcome to COQUI TTS")
    init_prompt()
    
if __name__ == "__main__":
    main()