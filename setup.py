#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
import sys

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py


parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to embed in wheel.')
parser.add_argument('--model_config', type=str, help='Path to model configuration file to embed in wheel.')
args, unknown_args = parser.parse_known_args()

# Remove our arguments from argv so that setuptools doesn't see them
sys.argv = [sys.argv[0]] + unknown_args

version = '0.0.1'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('TTS_PYTORCH_BUILD_VERSION'):
    version = os.getenv('TTS_PYTORCH_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):
    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


# The documentation for this feature is in server/README.md
package_data = ['server/templates/*']

if 'bdist_wheel' in unknown_args and args.checkpoint and args.model_config:
    print('Embedding model in wheel file...')
    model_dir = os.path.join('server', 'model')
    tts_dir = os.path.join(model_dir, 'tts')
    os.makedirs(tts_dir, exist_ok=True)
    embedded_checkpoint_path = os.path.join(tts_dir, 'checkpoint.pth.tar')
    shutil.copy(args.checkpoint, embedded_checkpoint_path)
    embedded_config_path = os.path.join(tts_dir, 'config.json')
    shutil.copy(args.model_config, embedded_config_path)
    package_data.extend([embedded_checkpoint_path, embedded_config_path])

setup(
    name='TTS',
    version=version,
    url='https://github.com/mozilla/TTS',
    description='Text to Speech with Deep Learning',
    license='MPL-2.0',
    package_dir={'': 'tts_namespace'},
    packages=find_packages('tts_namespace'),
    package_data={
        'TTS': package_data,
    },
    project_urls={
        'Documentation': 'https://github.com/mozilla/TTS/wiki',
        'Tracker': 'https://github.com/mozilla/TTS/issues',
        'Repository': 'https://github.com/mozilla/TTS',
        'Discussions': 'https://discourse.mozilla.org/c/tts',
    },
    cmdclass={
        'build_py': build_py,
        'develop': develop,
    },
    install_requires=[
        "scipy>=0.19.0",
        "torch>=0.4.1",
        "numpy==1.15.4",
        "librosa==0.6.2",
        "unidecode==0.4.20",
        "attrdict",
        "tensorboardX",
        "matplotlib",
        "Pillow",
        "flask",
        # "lws",
        "tqdm",
        "bokeh==1.4.0",
        "soundfile",
        "phonemizer @ https://github.com/bootphon/phonemizer/tarball/master",
    ],
    dependency_links=[
        "http://github.com/bootphon/phonemizer/tarball/master#egg=phonemizer-1.0.1"
    ]
)
