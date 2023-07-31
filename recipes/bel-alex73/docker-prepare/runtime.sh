#!/bin/bash

cd /a/TTS
pip install -e .[all,dev,notebooks]

LANG=C.utf8 bash
