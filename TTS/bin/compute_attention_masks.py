import argparse
import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import RawTextHelpFormatter
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.io import load_checkpoint
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Extract attention masks from trained Tacotron/Tacotron2 models.
These masks can be used for different purposes including training a TTS model with a Duration Predictor.\n\n'''

'''Each attention mask is written to the same path as the input wav file with ".npy" file extension.
(e.g. path/bla.wav (wav file) --> path/bla.npy (attention mask))\n'''

'''
Example run:
    CUDA_VISIBLE_DEVICE="0" python TTS/bin/compute_attention_masks.py
        --model_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/checkpoint_200000.pth.tar
        --config_path /data/rw/home/Models/ljspeech-dcattn-December-14-2020_11+10AM-9d0e8c7/config.json
        --dataset_metafile /root/LJSpeech-1.1/metadata.csv
        --data_path /root/LJSpeech-1.1/
        --batch_size 32
        --dataset ljspeech
        --use_cuda True
''',
        formatter_class=RawTextHelpFormatter
        )
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to Tacotron/Tacotron2 model file ')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to Tacotron/Tacotron2 config file.',
    )
    parser.add_argument('--dataset',
                        type=str,
                        default='',
                        required=True,
                        help='Target dataset processor name from TTS.tts.dataset.preprocess.')

    parser.add_argument(
        '--dataset_metafile',
        type=str,
        default='',
        required=True,
        help='Dataset metafile inclusing file paths with transcripts.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument('--use_cuda',
                        type=bool,
                        default=False,
                        help="enable/disable cuda.")

    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch size for the model. Use batch_size=1 if you have no CUDA.')
    args = parser.parse_args()

    C = load_config(args.config_path)
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    # TODO: handle multi-speaker
    model = setup_model(num_chars, num_speakers=0, c=C)
    model, _ = load_checkpoint(model, args.model_path, None, args.use_cuda)
    model.eval()

    # data loader
    preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
    preprocessor = getattr(preprocessor, args.dataset)
    meta_data = preprocessor(args.data_path, args.dataset_metafile)
    dataset = MyDataset(model.decoder.r,
                        C.text_cleaner,
                        compute_linear_spec=False,
                        ap=ap,
                        meta_data=meta_data,
                        tp=C.characters if 'characters' in C.keys() else None,
                        add_blank=C['add_blank'] if 'add_blank' in C.keys() else False,
                        use_phonemes=C.use_phonemes,
                        phoneme_cache_path=C.phoneme_cache_path,
                        phoneme_language=C.phoneme_language,
                        enable_eos_bos=C.enable_eos_bos_chars)

    dataset.sort_items()
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=4,
                        collate_fn=dataset.collate_fn,
                        shuffle=False,
                        drop_last=False)

    # compute attentions
    file_paths = []
    with torch.no_grad():
        for data in tqdm(loader):
            # setup input data
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[3]
            mel_input = data[4]
            mel_lengths = data[5]
            stop_targets = data[6]
            item_idxs = data[7]

            # dispatch data to GPU
            if args.use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            mel_outputs, postnet_outputs, alignments, stop_tokens = model.forward(
                text_input, text_lengths, mel_input)

            alignments = alignments.detach()
            for idx, alignment in enumerate(alignments):
                item_idx = item_idxs[idx]
                # interpolate if r > 1
                alignment = torch.nn.functional.interpolate(
                    alignment.transpose(0, 1).unsqueeze(0),
                    size=None,
                    scale_factor=model.decoder.r,
                    mode='nearest',
                    align_corners=None,
                    recompute_scale_factor=None).squeeze(0).transpose(0, 1)
                # remove paddings
                alignment = alignment[:mel_lengths[idx], :text_lengths[idx]].cpu().numpy()
                # set file paths
                wav_file_name = os.path.basename(item_idx)
                align_file_name = os.path.splitext(wav_file_name)[0] + '.npy'
                file_path = item_idx.replace(wav_file_name, align_file_name)
                # save output
                file_paths.append([item_idx, file_path])
                np.save(file_path, alignment)

        # ourput metafile
        metafile = os.path.join(args.data_path, "metadata_attn_mask.txt")

        with open(metafile, "w") as f:
            for p in file_paths:
                f.write(f"{p[0]}|{p[1]}\n")
        print(f" >> Metafile created: {metafile}")
