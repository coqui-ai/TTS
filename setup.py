#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess

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


def create_readme_rst():
    try:
        subprocess.check_call(
            [
                "pandoc", "--from=markdown", "--to=rst", "--output=README.rst",
                "README.md"
            ],
            cwd=cwd)
        print("Generated README.rst from README.md using pandoc.")
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


setup(
    name='TTS',
    version=version,
    url='https://github.com/mozilla/TTS',
    description='Text to Speech with Deep Learning',
    license='MPL-2.0',
    package_dir={'': 'tts_namespace'},
    packages=find_packages('tts_namespace'),
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
    setup_requires=["numpy==1.15.4"],
    install_requires=[
        "scipy >=0.19.0",
        "torch >= 0.4.1",
        "librosa==0.6.2",
        "unidecode==0.4.20",
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
