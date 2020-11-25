#!/usr/bin/env bash
set -xe
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
# run training
CUDA_VISIBLE_DEVICES="" python TTS/bin/train_tacotron.py --config_path $BASEDIR/inputs/test_train_config.json
# find the training folder
LATEST_FOLDER=$(ls $BASEDIR/train_outputs/| sort | tail -1)
echo $LATEST_FOLDER
# continue the previous training
CUDA_VISIBLE_DEVICES=""  python TTS/bin/train_tacotron.py --continue_path $BASEDIR/train_outputs/$LATEST_FOLDER
# remove all the outputs
rm -rf $BASEDIR/train_outputs/
