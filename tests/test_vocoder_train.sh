#!/usr/bin/env bash

BASEDIR=$(dirname "$0")
echo "$BASEDIR"
# create run dir
mkdir $BASEDIR/train_outputs
# run training
CUDA_VISIBLE_DEVICES="" python mozilla_voice_tts/bin/train_vocoder.py --config_path $BASEDIR/inputs/test_vocoder_multiband_melgan_config.json
# find the training folder
LATEST_FOLDER=$(ls $BASEDIR/outputs/train_outputs/| sort | tail -1)
echo $LATEST_FOLDER
# continue the previous training
CUDA_VISIBLE_DEVICES=""  python mozilla_voice_tts/bin/train_vocoder.py --continue_path $BASEDIR/outputs/train_outputs/$LATEST_FOLDER
# remove all the outputs
rm -rf $BASEDIR/train_outputs/
