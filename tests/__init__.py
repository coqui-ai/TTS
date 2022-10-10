import os

from TTS.config import BaseDatasetConfig
from TTS.utils.generic_utils import get_cuda


def get_device_id():
    use_cuda, _ = get_cuda()
    if use_cuda:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
            GPU_ID = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        else:
            GPU_ID = "0"
    else:
        GPU_ID = ""
    return GPU_ID


def get_tests_path():
    """Returns the path to the test directory."""
    return os.path.dirname(os.path.realpath(__file__))


def get_tests_input_path():
    """Returns the path to the test data directory."""
    return os.path.join(get_tests_path(), "inputs")


def get_tests_data_path():
    """Returns the path to the test data directory."""
    return os.path.join(get_tests_path(), "data")


def get_tests_output_path():
    """Returns the path to the directory for test outputs."""
    path = os.path.join(get_tests_path(), "outputs")
    os.makedirs(path, exist_ok=True)
    return path


def run_cli(command):
    exit_status = os.system(command)
    assert exit_status == 0, f" [!] command `{command}` failed."


def get_test_data_config():
    return BaseDatasetConfig(formatter="ljspeech", path="tests/data/ljspeech/", meta_file_train="metadata.csv")


def assertHasAttr(test_obj, obj, intendedAttr):
    # from https://stackoverflow.com/questions/48078636/pythons-unittest-lacks-an-asserthasattr-method-what-should-i-use-instead
    testBool = hasattr(obj, intendedAttr)
    test_obj.assertTrue(testBool, msg=f"obj lacking an attribute. obj: {obj}, intendedAttr: {intendedAttr}")


def assertHasNotAttr(test_obj, obj, intendedAttr):
    testBool = hasattr(obj, intendedAttr)
    test_obj.assertFalse(testBool, msg=f"obj should not have an attribute. obj: {obj}, intendedAttr: {intendedAttr}")
