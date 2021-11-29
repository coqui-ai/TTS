"""Get detailed info about the working environment."""
import os
import platform
import sys

import numpy
import torch

sys.path += [os.path.abspath(".."), os.path.abspath(".")]
import json

import TTS


def system_info():
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def cuda_info():
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
    }


def package_info():
    return {
        "numpy": numpy.__version__,
        "PyTorch_version": torch.__version__,
        "PyTorch_debug": torch.version.debug,
        "TTS": TTS.__version__,
    }


def main():
    details = {"System": system_info(), "CUDA": cuda_info(), "Packages": package_info()}
    print(json.dumps(details, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
