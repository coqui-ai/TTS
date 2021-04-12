#!/usr/bin/env bash
set -xe
BASEDIR=$(dirname "$0")
TARGET_SR=16000
echo "$BASEDIR"
#run the resample script
python TTS/bin/resample.py --input_dir $BASEDIR/../data/ljspeech --output_dir $BASEDIR/outputs/resample_tests --output_sr $TARGET_SR
#check samplerate of output
OUT_SR=$( (echo "import librosa" ; echo "y, sr = librosa.load('"$BASEDIR"/outputs/resample_tests/wavs/LJ001-0012.wav', sr=None)" ; echo "print(sr)") | python )
OUT_SR=$(($OUT_SR + 0))
if [[ $OUT_SR -ne $TARGET_SR ]]; then
    echo "Missmatch between target and output sample rates"
    exit 1
fi
#cleaning up
rm -rf $BASEDIR/outputs/resample_tests
