#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
import sys
import numpy

from setuptools import setup, find_packages, Extension
import setuptools.command.develop
import setuptools.command.build_py

# handle import if cython is not already installed.
try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize(*args, **kwargs):  #pylint: disable=redefined-outer-name
        from Cython.Build import cythonize  #pylint: disable=redefined-outer-name, import-outside-toplevel
        return cythonize(*args, **kwargs)


parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument('--checkpoint',
                    type=str,
                    help='Path to checkpoint file to embed in wheel.')
parser.add_argument('--model_config',
                    type=str,
                    help='Path to model configuration file to embed in wheel.')
args, unknown_args = parser.parse_known_args()

# Remove our arguments from argv so that setuptools doesn't see them
sys.argv = [sys.argv[0]] + unknown_args

version = '0.0.5'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('TTS_PYTORCH_BUILD_VERSION'):
    version = os.getenv('TTS_PYTORCH_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass
    except IOError:  # FileNotFoundError for python 3
        pass


# Handle Cython code
def find_pyx(path='.'):
    pyx_files = []
    for root, _, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files


def find_cython_extensions(path="."):
    exts = cythonize(find_pyx(path), language_level=3)
    for ext in exts:
        ext.include_dirs = [numpy.get_include()]
    return exts


class build_py(setuptools.command.build_py.build_py):  # pylint: disable=too-many-ancestors
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
package_data = ['TTS/server/templates/*']

if 'bdist_wheel' in unknown_args and args.checkpoint and args.model_config:
    print('Embedding model in wheel file...')
    model_dir = os.path.join('TTS', 'server', 'model')
    tts_dir = os.path.join(model_dir, 'tts')
    os.makedirs(tts_dir, exist_ok=True)
    embedded_checkpoint_path = os.path.join(tts_dir, 'checkpoint.pth.tar')
    shutil.copy(args.checkpoint, embedded_checkpoint_path)
    embedded_config_path = os.path.join(tts_dir, 'config.json')
    shutil.copy(args.model_config, embedded_config_path)
    package_data.extend([embedded_checkpoint_path, embedded_config_path])


def pip_install(package_name):
    subprocess.call([sys.executable, '-m', 'pip', 'install', package_name])


reqs_from_file = open('requirements.txt').readlines()
reqs_without_tf = [r for r in reqs_from_file if not r.startswith('tensorflow')]
tf_req = [r for r in reqs_from_file if r.startswith('tensorflow')]

requirements = {'install_requires': reqs_without_tf, 'pip_install': tf_req}

setup(
    name='TTS',
    version=version,
    url='https://github.com/mozilla/TTS',
    author='Eren Gölge',
    author_email='egolge@mozilla.com',
    description='Text to Speech with Deep Learning',
    license='MPL-2.0',
    entry_points={'console_scripts': ['tts-server = TTS.server.server:main']},
    ext_modules=find_cython_extensions(),
    packages=find_packages(include=['TTS*']),
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
    install_requires=requirements['install_requires'],
    python_requires='>=3.6.0',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research :: Developers",
        "Operating System :: POSIX :: Linux",
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        "Topic :: Software Development :: Libraries :: Python Modules :: Speech :: Sound/Audio :: Multimedia :: Artificial Intelligence",
    ])

# for some reason having tensorflow in 'install_requires'
# breaks some of the dependencies.
if 'bdist_wheel' not in unknown_args:
    for module in requirements['pip_install']:
        pip_install(module)
