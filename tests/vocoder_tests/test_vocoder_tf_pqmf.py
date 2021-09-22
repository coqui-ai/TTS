import os
import unittest

import soundfile as sf
import tensorflow as tf
import torch
from librosa.core import load

from tests import get_tests_input_path, get_tests_output_path, get_tests_path
from TTS.vocoder.tf.layers.pqmf import PQMF

TESTS_PATH = get_tests_path()
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")
use_cuda = torch.cuda.is_available()


@unittest.skipIf(use_cuda, " [!] Skip Test: Loosy TF support.")
def test_pqmf():
    w, sr = load(WAV_FILE)

    layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)
    w, sr = load(WAV_FILE)
    w2 = tf.convert_to_tensor(w[None, None, :])
    b2 = layer.analysis(w2)
    w2_ = layer.synthesis(b2)
    w2_ = w2.numpy()

    print(w2_.max())
    print(w2_.min())
    print(w2_.mean())
    sf.write(os.path.join(get_tests_output_path(), "tf_pqmf_output.wav"), w2_.flatten(), sr)
