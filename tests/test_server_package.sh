#!/bin/bash
set -xe

if [[ ! -f tests/outputs/checkpoint_10.pth.tar ]]; then
    echo "Missing dummy model in tests/outputs. This test needs to run after the Python unittests have been run."
    exit 1
fi

rm -f dist/*.whl
python3 setup.py --quiet bdist_wheel --checkpoint tests/outputs/checkpoint_10.pth.tar --model_config tests/outputs/dummy_model_config.json

python3 -m venv /tmp/venv
source /tmp/venv/bin/activate
python3 -m pip install --quiet --upgrade pip setuptools wheel cython
# wait to install numpy until we have wheel support
python3 -m pip install numpy
python3 -m pip install --quiet dist/TTS*.whl

# this is related to https://github.com/librosa/librosa/issues/1160
python3 -m pip install numba==0.48

python3 -m TTS.server.server &
SERVER_PID=$!

echo 'Waiting for server...'
sleep 30

curl -o /tmp/audio.wav "http://localhost:5002/api/tts?text=synthesis%20schmynthesis"
python3 -c 'import sys; import wave; print(wave.open(sys.argv[1]).getnframes())' /tmp/audio.wav

kill $SERVER_PID

deactivate
rm -rf /tmp/venv

rm /tmp/audio.wav
rm dist/*.whl
