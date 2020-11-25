set -e
TF_CPP_MIN_LOG_LEVEL=3

# tests
nosetests tests -x &&\

# runtime tests
./tests/test_server_package.sh && \
./tests/test_tts_train.sh && \
./tests/test_glow-tts_train.sh && \
./tests/test_vocoder_gan_train.sh && \
./tests/test_vocoder_wavernn_train.sh && \
./tests/test_vocoder_wavegrad_train.sh && \

# linter check
cardboardlinter --refspec master