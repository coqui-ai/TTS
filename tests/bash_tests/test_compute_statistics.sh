#!/usr/bin/env bash
set -xe
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
# run training
CUDA_VISIBLE_DEVICES="" python TTS/bin/compute_statistics.py --config_path $BASEDIR/../inputs/test_glow_tts.json --out_path $BASEDIR/../outputs/scale_stats.npy

