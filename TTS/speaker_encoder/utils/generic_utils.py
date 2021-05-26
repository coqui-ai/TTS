import datetime
import os
import re

import numpy as np
import torch
import glob
import random

from scipy import signal
from multiprocessing import Manager

from TTS.speaker_encoder.models.lstm import LSTMSpeakerEncoder
from TTS.speaker_encoder.models.resnet import ResNetSpeakerEncoder
from TTS.utils.generic_utils import check_argument

class Storage(object):
    def __init__(self, maxsize, storage_batchs, num_speakers_in_batch, num_threads=8):
        # use multiprocessing for threading safe
        self.storage = Manager().list()
        self.maxsize = maxsize
        self.num_speakers_in_batch = num_speakers_in_batch
        self.num_threads = num_threads
        self.ignore_last_batch = False

        if storage_batchs >= 3:
            self.ignore_last_batch = True

        # used for fast random sample
        self.safe_storage_size = self.maxsize - self.num_threads
        if self.ignore_last_batch:
            self.safe_storage_size -= self.num_speakers_in_batch

    def __len__(self):
        return len(self.storage)

    def full(self):
        return len(self.storage) >= self.maxsize

    def append(self, item):
        # if storage is full, remove an item
        if self.full():
            self.storage.pop(0)

        self.storage.append(item)

    def get_random_sample(self):
        # safe storage size considering all threads remove one item from storage in same time
        storage_size = len(self.storage) - self.num_threads

        if self.ignore_last_batch:
            storage_size -= self.num_speakers_in_batch

        return self.storage[random.randint(0, storage_size)]

    def get_random_sample_fast(self):
        '''Call this method only when storage is full'''
        return self.storage[random.randint(0, self.safe_storage_size)]

class AugmentWAV(object):

    def __init__(self, ap, augmentation_config):

        self.ap = ap
        self.use_additive_noise = False

        if 'additive' in augmentation_config.keys():
            self.additive_noise_config = augmentation_config['additive']
            additive_path = self.additive_noise_config['sounds_path']
            if additive_path:
                self.use_additive_noise = True
                # get noise types
                self.additive_noise_types = []
                for key in self.additive_noise_config.keys():
                    if isinstance(self.additive_noise_config[key], dict):
                        self.additive_noise_types.append(key)

                additive_files = glob.glob(os.path.join(additive_path, '**/*.wav'), recursive=True)

                self.noise_list = {}

                for wav_file in additive_files:
                    noise_dir = wav_file.replace(additive_path, '').split(os.sep)[0]
                    # ignore not listed directories
                    if noise_dir not in self.additive_noise_types:
                        continue
                    if not noise_dir in self.noise_list:
                        self.noise_list[noise_dir] = []
                    self.noise_list[noise_dir].append(wav_file)

                print(f" | > Using Additive Noise Augmentation: with {len(additive_files)} audios instances from {self.additive_noise_types}")

        self.use_rir = False

        if 'rir' in augmentation_config.keys():
            self.rir_config = augmentation_config['rir']
            if self.rir_config['rir_path']:
                self.rir_files = glob.glob(os.path.join(self.rir_config['rir_path'], '**/*.wav'), recursive=True)
                self.use_rir = True

            print(f" | > Using RIR Noise Augmentation: with {len(self.rir_files)} audios instances")

        self.create_augmentation_global_list()

    def create_augmentation_global_list(self):
        if self.use_additive_noise:
            self.global_noise_list = self.additive_noise_types
        else:
            self.global_noise_list = []
        if self.use_rir:
            self.global_noise_list.append("RIR_AUG")

    def additive_noise(self, noise_type, audio):

        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

        noise_list = random.sample(self.noise_list[noise_type], random.randint(self.additive_noise_config[noise_type]['min_num_noises'], self.additive_noise_config[noise_type]['max_num_noises']))

        audio_len = audio.shape[0]
        noises_wav = None
        for noise in noise_list:
            noiseaudio = self.ap.load_wav(noise, sr=self.ap.sample_rate)[:audio_len]

            if noiseaudio.shape[0] < audio_len:
                continue

            noise_snr = random.uniform(self.additive_noise_config[noise_type]['min_snr_in_db'], self.additive_noise_config[noise_type]['max_num_noises'])
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noise_wav = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio

            if noises_wav is None:
                noises_wav = noise_wav
            else:
                noises_wav += noise_wav

        # if all possible files is less than audio, choose other files
        if noises_wav is None:
            return self.additive_noise(noise_type, audio)

        return audio + noises_wav

    def reverberate(self, audio):
        audio_len = audio.shape[0]

        rir_file = random.choice(self.rir_files)
        rir = self.ap.load_wav(rir_file, sr=self.ap.sample_rate)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode=self.rir_config['conv_mode'])[:audio_len]

    def apply_one(self, audio):
        noise_type = random.choice(self.global_noise_list)
        if noise_type == "RIR_AUG":
            return self.reverberate(audio)

        return self.additive_noise(noise_type, audio)

def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)

def setup_model(c):
    if c.model_name.lower() == 'lstm':
        model = LSTMSpeakerEncoder(c.model["input_dim"], c.model["proj_dim"], c.model["lstm_dim"], c.model["num_lstm_layers"])
    elif c.model_name.lower() == 'resnet':
        model = ResNetSpeakerEncoder(input_dim=c.model["input_dim"], proj_dim=c.model["proj_dim"])
    return model


def save_checkpoint(model, optimizer, criterion, model_loss, out_path, current_step, epoch):
    checkpoint_path = "checkpoint_{}.pth.tar".format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        "model": new_state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "criterion": criterion.state_dict(),
        "step": current_step,
        "epoch": epoch,
        "loss": model_loss,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    torch.save(state, checkpoint_path)


def save_best_model(model, optimizer, criterion, model_loss, best_loss, out_path, current_step):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            "model": new_state_dict,
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "step": current_step,
            "loss": model_loss,
            "date": datetime.date.today().strftime("%B %d, %Y"),
        }
        best_loss = model_loss
        bestmodel_path = "best_model.pth.tar"
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(model_loss, bestmodel_path))
        torch.save(state, bestmodel_path)
    return best_loss


def check_config_speaker_encoder(c):
    """Check the config.json file of the speaker encoder"""
    check_argument("run_name", c, restricted=True, val_type=str)
    check_argument("run_description", c, val_type=str)

    # audio processing parameters
    check_argument("audio", c, restricted=True, val_type=dict)
    check_argument("num_mels", c["audio"], restricted=True, val_type=int, min_val=10, max_val=2056)
    check_argument("fft_size", c["audio"], restricted=True, val_type=int, min_val=128, max_val=4058)
    check_argument("sample_rate", c["audio"], restricted=True, val_type=int, min_val=512, max_val=100000)
    check_argument(
        "frame_length_ms",
        c["audio"],
        restricted=True,
        val_type=float,
        min_val=10,
        max_val=1000,
        alternative="win_length",
    )
    check_argument(
        "frame_shift_ms", c["audio"], restricted=True, val_type=float, min_val=1, max_val=1000, alternative="hop_length"
    )
    check_argument("preemphasis", c["audio"], restricted=True, val_type=float, min_val=0, max_val=1)
    check_argument("min_level_db", c["audio"], restricted=True, val_type=int, min_val=-1000, max_val=10)
    check_argument("ref_level_db", c["audio"], restricted=True, val_type=int, min_val=0, max_val=1000)
    check_argument("power", c["audio"], restricted=True, val_type=float, min_val=1, max_val=5)
    check_argument("griffin_lim_iters", c["audio"], restricted=True, val_type=int, min_val=10, max_val=1000)

    # training parameters
    check_argument("loss", c, enum_list=["ge2e", "angleproto", "softmaxproto"], restricted=True, val_type=str)
    check_argument("grad_clip", c, restricted=True, val_type=float)
    check_argument("epochs", c, restricted=True, val_type=int, min_val=1)
    check_argument("lr", c, restricted=True, val_type=float, min_val=0)
    check_argument("lr_decay", c, restricted=True, val_type=bool)
    check_argument("warmup_steps", c, restricted=True, val_type=int, min_val=0)
    check_argument("tb_model_param_stats", c, restricted=True, val_type=bool)
    check_argument("num_speakers_in_batch", c, restricted=True, val_type=int)
    check_argument("num_loader_workers", c, restricted=True, val_type=int)
    check_argument("wd", c, restricted=True, val_type=float, min_val=0.0, max_val=1.0)

    # checkpoint and output parameters
    check_argument("steps_plot_stats", c, restricted=True, val_type=int)
    check_argument("checkpoint", c, restricted=True, val_type=bool)
    check_argument("save_step", c, restricted=True, val_type=int)
    check_argument("print_step", c, restricted=True, val_type=int)
    check_argument("output_path", c, restricted=True, val_type=str)

    # model parameters
    check_argument("model", c, restricted=True, val_type=dict)
    check_argument("model_name", c, restricted=True, val_type=str)
    check_argument("input_dim", c["model"], restricted=True, val_type=int)
    check_argument("proj_dim", c["model"], restricted=True, val_type=int)

    if c.model_name.lower() == 'lstm':
        check_argument("lstm_dim", c["model"], restricted=True, val_type=int)
        check_argument("num_lstm_layers", c["model"], restricted=True, val_type=int)
        check_argument("use_lstm_with_projection", c["model"], restricted=True, val_type=bool)

    # in-memory storage parameters
    check_argument("storage", c, restricted=True, val_type=dict)
    check_argument("sample_from_storage_p", c["storage"], restricted=True, val_type=float, min_val=0.0, max_val=1.0)
    check_argument("storage_size", c["storage"], restricted=True, val_type=int, min_val=1, max_val=100)

    # datasets - checking only the first entry
    check_argument("datasets", c, restricted=True, val_type=list)
    for dataset_entry in c["datasets"]:
        check_argument("name", dataset_entry, restricted=True, val_type=str)
        check_argument("path", dataset_entry, restricted=True, val_type=str)
        check_argument("meta_file_train", dataset_entry, restricted=True, val_type=[str, list])
        check_argument("meta_file_val", dataset_entry, restricted=True, val_type=str)
