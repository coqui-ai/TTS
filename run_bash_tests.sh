set -e
TF_CPP_MIN_LOG_LEVEL=3

# runtime bash based tests
./tests/bash_tests/test_demo_server.sh && \
./tests/bash_tests/test_resample.sh && \
./tests/bash_tests/test_tacotron_train.sh && \
./tests/bash_tests/test_glow-tts_train.sh && \
./tests/bash_tests/test_vocoder_gan_train.sh && \
./tests/bash_tests/test_vocoder_wavernn_train.sh && \
./tests/bash_tests/test_vocoder_wavegrad_train.sh && \
./tests/bash_tests/test_speedy_speech_train.sh && \
./tests/bash_tests/test_aligntts_train.sh && \
./tests/bash_tests/test_compute_statistics.sh
