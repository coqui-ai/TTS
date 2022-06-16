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

def extract_aligments(
    data_loader, model, output_path, use_cuda=True
):
    model.eval()
    export_metadata = []
    for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        batch = model.format_batch(batch)        
        if use_cuda:
            for k, v in batch.items():
                    batch[k] = to_cuda(v)

        batch = model.format_batch_on_device(batch)

        spec_lens = batch["spec_lens"]
        tokens = batch["tokens"]
        token_lenghts = batch["token_lens"]
        spec = batch["spec"]

        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        language_ids = batch["language_ids"]
        emotion_embeddings = batch["emotion_embeddings"]
        emotion_ids = batch["emotion_ids"]
        waveform = batch["waveform"]
        item_idx = batch["audio_files"]
        # generator pass
        outputs = model.forward(
            tokens,
            token_lenghts,
            spec,
            spec_lens,
            waveform,
            aux_input={
                "d_vectors": d_vectors,
                "speaker_ids": speaker_ids,
                "language_ids": language_ids,
                "emotion_embeddings": emotion_embeddings,
                "emotion_ids": emotion_ids,
            },
        )

        alignments = outputs["alignments"].detach().cpu().numpy()

        for idx in range(tokens.shape[0]):
            wav_file_path = item_idx[idx]
            alignment = alignments[idx]
            # set paths
            align_file_name = os.path.splitext(os.path.basename(wav_file_path))[0] + ".npy"
            os.makedirs(os.path.join(output_path, "alignments"), exist_ok=True)
            align_file_path = os.path.join(output_path, "alignments", align_file_name)
            np.save(align_file_path, alignment)


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data, speaker_manager

    # load data instances
    meta_data_train, meta_data_eval = load_tts_samples(
        c.datasets, eval_split=args.eval, eval_split_max_size=c.eval_split_max_size, eval_split_size=c.eval_split_size
    )

    # use eval and training partitions
    meta_data = meta_data_train + meta_data_eval

    # setup model
    model = setup_model(c, meta_data)

    # restore model
    model.load_checkpoint(c, args.checkpoint_path, eval=False)
    model = model.eval()

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

    extract_aligments(
        own_loader,
        model,
        args.output_path,
        use_cuda=use_cuda,
    )


if __name__ == "__main__":
    # python3 TTS/bin/extract_tts_audio.py --config_path /raid/edresson/dev/Checkpoints/YourTTS/new_vctk_trimmed_silence/upsampling/YourTTS_22khz--\>44khz_vocoder_approach_frozen/YourTTS_22khz--\>44khz_vocoder_approach_frozen-April-02-2022_08+23PM-a5f5ebae/config.json --checkpoint_path /raid/edresson/dev/Checkpoints/YourTTS/new_vctk_trimmed_silence/upsampling/YourTTS_22khz--\>44khz_vocoder_approach_frozen/YourTTS_22khz--\>44khz_vocoder_approach_frozen-April-02-2022_08+23PM-a5f5ebae/checkpoint_1600000.pth --output_path ../Test_extract_audio_script/
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file for training.", required=True)
    parser.add_argument("--checkpoint_path", type=str, help="Model file to be restored.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save mel specs", required=True)
    parser.add_argument("--eval", type=bool, help="compute eval.", default=True)
    args = parser.parse_args()

    c = load_config(args.config_path)
    # disable samplers
    c.use_speaker_weighted_sampler = False
    c.use_language_weighted_sampler = False
    c.use_length_weighted_sampler = False
    c.use_style_weighted_sampler = False
    main(args)