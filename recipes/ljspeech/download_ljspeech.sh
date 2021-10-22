#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download LJSpeech dataset
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# extract
tar -xjf LJSpeech-1.1.tar.bz2
# create train-val splits
shuf LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_shuf.csv
head -n 12000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
tail -n 1100 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv
mv LJSpeech-1.1 $RUN_DIR/recipes/ljspeech/
rm LJSpeech-1.1.tar.bz2