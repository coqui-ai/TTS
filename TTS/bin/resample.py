import argparse
import glob
import os
import librosa
from distutils.dir_util import copy_tree
from argparse import RawTextHelpFormatter
from multiprocessing import Pool
from tqdm import tqdm

def resample_file(func_args):
    filename, output_sr = func_args
    y, sr = librosa.load(filename, sr=output_sr)
    librosa.output.write_wav(filename, y, sr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''Resample a folder recusively with librosa
                       Can be used in place or create a copy of the folder as an output.\n\n
                       Example run:
                            python TTS/bin/resample.py
                                --input_dir /root/LJSpeech-1.1/
                                --output_sr 22050
                                --output_dir /root/resampled_LJSpeech-1.1/
                                --n_jobs 24
                    ''',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('--input_dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Path of the folder containing the audio files to resample')

    parser.add_argument('--output_sr',
                        type=int,
                        default=22050,
                        required=False,
                        help='Samlple rate to which the audio files should be resampled')

    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        required=False,
                        help='Path of the destination folder. If not defined, the operation is done in place')

    parser.add_argument('--n_jobs',
                        type=int,
                        default=None,
                        help='Number of threads to use, by default it uses all cores')

    args = parser.parse_args()

    if args.output_dir:
        print('Recursively copying the input folder...')
        copy_tree(args.input_dir, args.output_dir)
        args.input_dir = args.output_dir

    print('Resampling the audio files...')
    audio_files = glob.glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True)
    print(f'Found {len(audio_files)} files...')
    audio_files = list(zip(audio_files, len(audio_files)*[args.output_sr]))
    with Pool(processes=args.n_jobs) as p:
        with tqdm(total=len(audio_files)) as pbar:
            for i, _ in enumerate(p.imap_unordered(resample_file, audio_files)):
                pbar.update()

    print('Done !')
