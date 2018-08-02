'''
Extract spectrograms and save them to file for training
'''
import os
import sys
import time
import glob
import argparse
import librosa
import numpy as np
import tqdm
from utils.generic_utils import load_config

from multiprocessing import Pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Data folder.')
    parser.add_argument('--out_path', type=str, help='Output folder.')
    parser.add_argument(
        '--config', type=str, help='conf.json file for run settings.')
    parser.add_argument(
        "--num_proc", type=int, default=8, help="number of processes.")
    parser.add_argument(
        "--trim_silence",
        type=bool,
        default=False,
        help="trim silence in the voice clip.")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    OUT_PATH = args.out_path
    CONFIG = load_config(args.config)

    print(" > Input path: ", DATA_PATH)
    print(" > Output path: ", OUT_PATH)

    audio = importlib.import_module('utils.' + c.audio_processor)
    AudioProcessor = getattr(audio, 'AudioProcessor')
    ap = AudioProcessor(
        sample_rate=CONFIG.sample_rate,
        num_mels=CONFIG.num_mels,
        min_level_db=CONFIG.min_level_db,
        frame_shift_ms=CONFIG.frame_shift_ms,
        frame_length_ms=CONFIG.frame_length_ms,
        ref_level_db=CONFIG.ref_level_db,
        num_freq=CONFIG.num_freq,
        power=CONFIG.power,
        preemphasis=CONFIG.preemphasis,
        min_mel_freq=CONFIG.min_mel_freq,
        max_mel_freq=CONFIG.max_mel_freq)

    def trim_silence(self, wav):
        margin = int(CONFIG.sample_rate * 0.1)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=40, frame_length=1024, hop_length=256)[0]

    def extract_mel(file_path):
        # x, fs = sf.read(file_path)
        x, fs = librosa.load(file_path, CONFIG.sample_rate)
        if args.trim_silence:
            x = trim_silence(x)
        mel = ap.melspectrogram(x.astype('float32')).astype('float32')
        linear = ap.spectrogram(x.astype('float32')).astype('float32')
        file_name = os.path.basename(file_path).replace(".wav", "")
        mel_file = file_name + ".mel"
        linear_file = file_name + ".linear"
        np.save(os.path.join(OUT_PATH, mel_file), mel, allow_pickle=False)
        np.save(
            os.path.join(OUT_PATH, linear_file), linear, allow_pickle=False)
        mel_len = mel.shape[1]
        linear_len = linear.shape[1]
        wav_len = x.shape[0]
        print(" > " + file_path, flush=True)
        return file_path, mel_file, linear_file, str(wav_len), str(
            mel_len), str(linear_len)

    glob_path = os.path.join(DATA_PATH, "*.wav")
    print(" > Reading wav: {}".format(glob_path))
    file_names = glob.glob(glob_path, recursive=True)

    if __name__ == "__main__":
        print(" > Number of files: %i" % (len(file_names)))
        if not os.path.exists(OUT_PATH):
            os.makedirs(OUT_PATH)
            print(" > A new folder created at {}".format(OUT_PATH))

        r = []
        if args.num_proc > 1:
            print(" > Using {} processes.".format(args.num_proc))
            with Pool(args.num_proc) as p:
                r = list(
                    tqdm.tqdm(
                        p.imap(extract_mel, file_names),
                        total=len(file_names)))
                # r = list(p.imap(extract_mel, file_names))
        else:
            print(" > Using single process run.")
            for file_name in file_names:
                print(" > ", file_name)
                r.append(extract_mel(file_name))

        file_path = os.path.join(OUT_PATH, "meta_fftnet.csv")
        file = open(file_path, "w")
        for line in r:
            line = ", ".join(line)
            file.write(line + '\n')
        file.close()
