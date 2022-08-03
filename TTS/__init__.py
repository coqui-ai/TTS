import os

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r", encoding="utf-8") as f:
    version = f.read().strip()

__version__ = version
