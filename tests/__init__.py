import os


def get_tests_path():
    """Returns the path to the test directory."""
    return os.path.dirname(os.path.realpath(__file__))


def get_tests_input_path():
    """Returns the path to the test data directory."""
    return os.path.join(get_tests_path(), "inputs")


def get_tests_output_path():
    """Returns the path to the directory for test outputs."""
    return os.path.join(get_tests_path(), "outputs")
