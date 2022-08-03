set -e
TF_CPP_MIN_LOG_LEVEL=3

# runtime bash based tests
# TODO: move these to python
./tests/bash_tests/test_demo_server.sh && \
./tests/bash_tests/test_compute_statistics.sh
