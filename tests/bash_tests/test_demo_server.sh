#!/bin/bash
set -xe

python -m TTS.server.server &
SERVER_PID=$!

echo 'Waiting for server...'
sleep 30

curl -o /tmp/audio.wav "http://localhost:5002/api/tts?text=synthesis%20schmynthesis"
python -c 'import sys; import wave; print(wave.open(sys.argv[1]).getnframes())' /tmp/audio.wav

kill $SERVER_PID

rm /tmp/audio.wav
