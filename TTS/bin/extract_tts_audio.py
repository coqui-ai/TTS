#!/usr/bin/env python3
"""Extract Mel spectrograms with teacher forcing."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import count_parameters
from trainer.generic_utils import to_cuda

use_cuda = torch.cuda.is_available()

def set_filename(wav_path, out_path):
    wav_file = os.path.basename(wav_path)
    file_name = wav_file.split(".")[0]
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav_gt"), exist_ok=True)
    wavq_path = os.path.join(out_path, "quant", file_name)
    wav_gt_path = os.path.join(out_path, "wav_gt", file_name + ".wav")
    wav_path = os.path.join(out_path, "wav", file_name + ".wav")
    return file_name, wavq_path, wav_gt_path, wav_path


def extract_audios(
    data_loader, model, ap, output_path, quantized_wav=False, save_gt_audio=False, use_cuda=True
):
    model.eval()
    export_metadata = []
    for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        batch = model.format_batch(batch)
        batch = model.format_batch_on_device(batch)
        
        if use_cuda:
            for k, v in batch.items():
                    batch[k] = to_cuda(v)

        tokens = batch["tokens"]
        token_lenghts = batch["token_lens"]
        spec = batch["spec"]
        spec_lens = batch["spec_lens"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        language_ids = batch["language_ids"]
        item_idx = batch["audio_files_path"]
        wav_lengths = batch["waveform_lens"]

        outputs = model.inference_with_MAS(
            tokens,
            spec,
            spec_lens,
            aux_input={"x_lengths": token_lenghts, "d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
        )

        model_output = outputs["model_outputs"]
        model_output = model_output.detach().cpu().numpy()

        for idx in range(tokens.shape[0]):
            wav_file_path = item_idx[idx]
            wav_gt = ap.load_wav(wav_file_path)
            
            _, wavq_path, wav_gt_path, wav_path = set_filename(wav_file_path, output_path)

            # quantize and save wav
            if quantized_wav:
                wavq = ap.quantize(wav_gt)
                np.save(wavq_path, wavq)

            # save TTS mel
            wav = model_output[idx][0]
            wav_length = wav_lengths[idx]
            wav = wav[:wav_length]
            ap.save_wav(wav, wav_path)

            if save_gt_audio:
                ap.save_wav(wav_gt, wav_gt_path)


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data, speaker_manager

    # Audio processor
    ap = AudioProcessor(**c.audio)

    # load data instances
    meta_data_train, meta_data_eval = load_tts_samples(
        c.datasets, eval_split=args.eval, eval_split_max_size=c.eval_split_max_size, eval_split_size=c.eval_split_size
    )

    # use eval and training partitions
    meta_data = meta_data_train + meta_data_eval

    # setup model
    model = setup_model(c, meta_data)

    # restore model
    model.load_checkpoint(c, args.checkpoint_path, eval=True)

    if use_cuda:
        model.cuda()

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    own_loader = model.get_data_loader(config=model.config,
        assets={},
        is_eval=False,
        samples=meta_data,
        verbose=True,
        num_gpus=1,
    )

    extract_audios(
        own_loader,
        model,
        ap,
        args.output_path,
        quantized_wav=args.quantized,
        save_gt_audio=args.save_gt_audio,
        use_cuda=use_cuda,
    )


if __name__ == "__main__":
    # python3 TTS/bin/extract_tts_audio.py --config_path /raid/edresson/dev/Checkpoints/YourTTS/new_vctk_trimmed_silence/upsampling/YourTTS_22khz--\>44khz_vocoder_approach_frozen/YourTTS_22khz--\>44khz_vocoder_approach_frozen-April-02-2022_08+23PM-a5f5ebae/config.json --checkpoint_path /raid/edresson/dev/Checkpoints/YourTTS/new_vctk_trimmed_silence/upsampling/YourTTS_22khz--\>44khz_vocoder_approach_frozen/YourTTS_22khz--\>44khz_vocoder_approach_frozen-April-02-2022_08+23PM-a5f5ebae/checkpoint_1600000.pth --output_path ../Test_extract_audio_script/
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file for training.", required=True)
    parser.add_argument("--checkpoint_path", type=str, help="Model file to be restored.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save mel specs", required=True)
    parser.add_argument("--save_gt_audio", default=False, action="store_true", help="Save audio files")
    parser.add_argument("--quantized", action="store_true", help="Save quantized audio files")
    parser.add_argument("--eval", type=bool, help="compute eval.", default=True)
    args = parser.parse_args()

    c = load_config(args.config_path)
    c.audio.trim_silence = False
    c.batch_size = 4
    main(args)
