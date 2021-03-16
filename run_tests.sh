set -e
TF_CPP_MIN_LOG_LEVEL=3

# tests
nosetests tests -x &&\

# runtime tests
./tests/test_demo_server.sh && \
./tests/test_tacotron_train.sh && \
./tests/test_glow-tts_train.sh && \
./tests/test_vocoder_gan_train.sh && \
./tests/test_vocoder_wavernn_train.sh && \
./tests/test_vocoder_wavegrad_train.sh && \
./tests/test_speedy_speech_train.sh && \
./tests/test_compute_statistics.sh && \

# linter check
cardboardlinter --refspec main