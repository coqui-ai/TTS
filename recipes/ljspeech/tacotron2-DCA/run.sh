#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# # download LJSpeech dataset
# wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# # extract
# tar -xjf LJSpeech-1.1.tar.bz2
# # create train-val splits
# shuf LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_shuf.csv
# head -n 12000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
# tail -n 1100 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv
# mv LJSpeech-1.1 $RUN_DIR/
# rm LJSpeech-1.1.tar.bz2
# # compute dataset mean and variance for normalization
# python TTS/bin/compute_statistics.py $RUN_DIR/tacotron2-DDC.json $RUN_DIR/scale_stats.npy --data_path $RUN_DIR/LJSpeech-1.1/wavs/
# training ....
# change the GPU id if needed
CUDA_VISIBLE_DEVICES="0" python TTS/bin/train_tts.py --config_path $RUN_DIR/tacotron2-DCA.json \
                                                     --coqpit.output_path $RUN_DIR  \
                                                     --coqpit.datasets.0.path /media/erogol/nvme_linux/gdrive/Projects/TTS/recipes/ljspeech/tacotron2-DDC/LJSpeech-1.1/    \
                                                     --coqpit.audio.stats_path $RUN_DIR/scale_stats.npy \