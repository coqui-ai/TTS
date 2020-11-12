#!/usr/bin/env bash
set -xe
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
# run training
CUDA_VISIBLE_DEVICES="" python TTS/bin/train_glow_tts.py --config_path $BASEDIR/inputs/test_glow_tts.json
# find the training folder
LATEST_FOLDER=$(ls $BASEDIR/train_outputs/| sort | tail -1)
echo $LATEST_FOLDER
# continue the previous training
CUDA_VISIBLE_DEVICES=""  python TTS/bin/train_glow_tts.py --continue_path $BASEDIR/train_outputs/$LATEST_FOLDER
# remove all the outputs
rm -rf $BASEDIR/train_outputs/
