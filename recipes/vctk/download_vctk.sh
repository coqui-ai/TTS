#!/usr/bin/env bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR
# download VCTK dataset
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip -O VCTK-Corpus-0.92.zip
# extract
mkdir VCTK
unzip VCTK-Corpus-0.92 -d VCTK
# create train-val splits
mv VCTK $RUN_DIR/recipes/vctk/
rm VCTK-Corpus-0.92.zip
