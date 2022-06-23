#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from inquirer.themes import GreenPassion
import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

import inquirer

path = r"C:\Users\User\Desktop\TTS\TTS\.models.json"
manager = ModelManager(path)


def official_zoo_inquirer():
    model_list=manager.list_models(print_list=False)
    model_list['vocoder_models'].append('default_vocoder')
    model_load_questions = [
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

def multispeaker_inquirer():
    multispeaker_questions = [
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
                default="Enter some text."
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
                            'continue new text',
                            'restart tts',
                            'exit tts'
                        ]
                    ),
        ]

    continue_answers = inquirer.prompt(continue_questions, theme=GreenPassion())
    return continue_answers

def tts_inquirer(
    synthesizer,
    text,
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
    text = answers_tts['text_input']
    outpath = answers_tts['out_path']
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

def init_prompt():
    model_path=None
    config_path=None
    speakers_file_path=None
    language_ids_file_path=None
    vocoder_path=None
    vocoder_config_path=None
    encoder_path=None
    encoder_config_path=None
    use_cuda=True

    text="Random Text."
    speaker_idx=None
    language_idx=None
    speaker_wav=None
    reference_wav=reference_wav=None
    reference_speaker_idx=reference_speaker_name=None
    capacitron_style_wav=style_wav=None
    capacitron_style_text=style_text=None
      
    questions = [
    inquirer.List('to_do',
                    message="What do you need?",
                    choices=[
                        'play with your own model', 
                        'play with official model zoo',
                        'exit tts'
                    ]
                ),
    ]

    answers = inquirer.prompt(questions, theme=GreenPassion())

    if answers['to_do'] == 'exit tts':
        return

    if answers['to_do'] == 'play with official model zoo':
        answers_model_load = official_zoo_inquirer()
        tts_model_name=answers_model_load['tts_choose']
        model_path, config_path, model_item = manager.download_model(tts_model_name)
        vocoder_name=answers_model_load['vocoder_choose'] if answers_model_load['vocoder_choose'] != "default_vocoder" else model_item["default_vocoder"]  
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
        
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
     
        if synthesizer.tts_speakers_file:
            answers_multispeaker=multispeaker_inquirer()
            speaker_idx=answers_multispeaker['speaker_idx'],
            language_idx=answers_multispeaker['language_idx'],
            speaker_wav=answers_multispeaker['speaker_wav'],
            reference_wav=reference_wav=answers_multispeaker['reference_wav'],
            reference_speaker_idx=answers_multispeaker['reference_speaker_idx'],
    
        if 'capacitron' in tts_model_name:
            answers_capacitron = capacitron_inquirer()
            capacitron_style_wav=answers_capacitron['capacitron_style_wav'],
            capacitron_style_text=answers_capacitron['capacitron_style_text'],
        
        continue_answers = tts_inquirer(
                synthesizer,
                text,
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
            init_prompt()
        if continue_answers['to_do'] == 'continue new text':
            tts_inquirer(
                synthesizer,
                text,
                speaker_idx,
                language_idx,
                speaker_wav,
                reference_wav,
                reference_speaker_idx,
                capacitron_style_wav,
                capacitron_style_text
            )
        

init_prompt()