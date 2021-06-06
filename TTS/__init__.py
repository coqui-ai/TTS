import os

with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
    version = f.read().strip()

__version__ = version
