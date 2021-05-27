#!/usr/bin/env python3
"""Extract Mel spectrograms with teacher forcing."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.speakers import parse_speakers
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import count_parameters

use_cuda = torch.cuda.is_available()


def setup_loader(ap, r, verbose=False):
    dataset = MyDataset(
        r,
        c.text_cleaner,
        compute_linear_spec=False,
        meta_data=meta_data,
        ap=ap,
        tp=c.characters if "characters" in c.keys() else None,
        add_blank=c["add_blank"] if "add_blank" in c.keys() else False,
        batch_group_size=0,
        min_seq_len=c.min_seq_len,
        max_seq_len=c.max_seq_len,
        phoneme_cache_path=c.phoneme_cache_path,
        use_phonemes=c.use_phonemes,
        phoneme_language=c.phoneme_language,
        enable_eos_bos=c.enable_eos_bos_chars,
        use_noise_augment=False,
        verbose=verbose,
        speaker_mapping=speaker_mapping if c.use_speaker_embedding and c.use_external_speaker_embedding_file else None,
    )

    if c.use_phonemes and c.compute_input_seq_cache:
        # precompute phonemes to have a better estimate of sequence lengths.
        dataset.compute_input_seq(c.num_loader_workers)
    dataset.sort_items()

    loader = DataLoader(
        dataset,
        batch_size=c.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
        sampler=None,
        num_workers=c.num_loader_workers,
        pin_memory=False,
    )
    return loader


def set_filename(wav_path, out_path):
    wav_file = os.path.basename(wav_path)
    file_name = wav_file.split(".")[0]
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav_gl"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav"), exist_ok=True)
    wavq_path = os.path.join(out_path, "quant", file_name)
    mel_path = os.path.join(out_path, "mel", file_name)
    wav_gl_path = os.path.join(out_path, "wav_gl", file_name + ".wav")
    wav_path = os.path.join(out_path, "wav", file_name + ".wav")
    return file_name, wavq_path, mel_path, wav_gl_path, wav_path


def format_data(data):
    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    mel_input = data[4]
    mel_lengths = data[5]
    item_idx = data[7]
    attn_mask = data[9]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if c.use_speaker_embedding:
        if c.use_external_speaker_embedding_file:
            speaker_embeddings = data[8]
            speaker_ids = None
        else:
            speaker_ids = [speaker_mapping[speaker_name] for speaker_name in speaker_names]
            speaker_ids = torch.LongTensor(speaker_ids)
            speaker_embeddings = None
    else:
        speaker_embeddings = None
        speaker_ids = None

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.cuda(non_blocking=True)

        if attn_mask is not None:
            attn_mask = attn_mask.cuda(non_blocking=True)
    return (
        text_input,
        text_lengths,
        mel_input,
        mel_lengths,
        speaker_ids,
        speaker_embeddings,
        avg_text_length,
        avg_spec_length,
        attn_mask,
        item_idx,
    )


@torch.no_grad()
def inference(
    model_name,
    model,
    ap,
    text_input,
    text_lengths,
    mel_input,
    mel_lengths,
    attn_mask=None,
    speaker_ids=None,
    speaker_embeddings=None,
):
    if model_name == "glow_tts":
        mel_input = mel_input.permute(0, 2, 1)  # B x D x T
        speaker_c = None
        if speaker_ids is not None:
            speaker_c = speaker_ids
        elif speaker_embeddings is not None:
            speaker_c = speaker_embeddings

        model_output, *_ = model.inference_with_MAS(
            text_input, text_lengths, mel_input, mel_lengths, attn_mask, g=speaker_c
        )
        model_output = model_output.transpose(1, 2).detach().cpu().numpy()

    elif "tacotron" in model_name:
        _, postnet_outputs, *_ = model(
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            speaker_ids=speaker_ids,
            speaker_embeddings=speaker_embeddings,
        )
        # normalize tacotron output
        if model_name == "tacotron":
            mel_specs = []
            postnet_outputs = postnet_outputs.data.cpu().numpy()
            for b in range(postnet_outputs.shape[0]):
                postnet_output = postnet_outputs[b]
                mel_specs.append(torch.FloatTensor(ap.out_linear_to_mel(postnet_output.T).T))
            model_output = torch.stack(mel_specs).cpu().numpy()

        elif model_name == "tacotron2":
            model_output = postnet_outputs.detach().cpu().numpy()
    return model_output


def extract_spectrograms(
    data_loader, model, ap, output_path, quantized_wav=False, save_audio=False, debug=False, metada_name="metada.txt"
):
    model.eval()
    export_metadata = []
    for _, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        # format data
        (
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            speaker_ids,
            speaker_embeddings,
            _,
            _,
            attn_mask,
            item_idx,
        ) = format_data(data)

        model_output = inference(
            c.model.lower(),
            model,
            ap,
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            attn_mask,
            speaker_ids,
            speaker_embeddings,
        )

        for idx in range(text_input.shape[0]):
            wav_file_path = item_idx[idx]
            wav = ap.load_wav(wav_file_path)
            _, wavq_path, mel_path, wav_gl_path, wav_path = set_filename(wav_file_path, output_path)

            # quantize and save wav
            if quantized_wav:
                wavq = ap.quantize(wav)
                np.save(wavq_path, wavq)

            # save TTS mel
            mel = model_output[idx]
            mel_length = mel_lengths[idx]
            mel = mel[:mel_length, :].T
            np.save(mel_path, mel)

            export_metadata.append([wav_file_path, mel_path])
            if save_audio:
                ap.save_wav(wav, wav_path)

            if debug:
                print("Audio for debug saved at:", wav_gl_path)
                wav = ap.inv_melspectrogram(mel)
                ap.save_wav(wav, wav_gl_path)

    with open(os.path.join(output_path, metada_name), "w") as f:
        for data in export_metadata:
            f.write(f"{data[0]}|{data[1]+'.npy'}\n")


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data, symbols, phonemes, model_characters, speaker_mapping

    # Audio processor
    ap = AudioProcessor(**c.audio)
    if "characters" in c.keys() and c["characters"]:
        symbols, phonemes = make_symbols(**c.characters)

    # set model characters
    model_characters = phonemes if c.use_phonemes else symbols
    num_chars = len(model_characters)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    # use eval and training partitions
    meta_data = meta_data_train + meta_data_eval

    # parse speakers
    num_speakers, speaker_embedding_dim, speaker_mapping = parse_speakers(c, args, meta_data_train, None)

    # setup model
    model = setup_model(num_chars, num_speakers, c, speaker_embedding_dim=speaker_embedding_dim)

    # restore model
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    if use_cuda:
        model.cuda()

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)
    # set r
    r = 1 if c.model.lower() == "glow_tts" else model.decoder.r
    own_loader = setup_loader(ap, r, verbose=True)

    extract_spectrograms(
        own_loader,
        model,
        ap,
        args.output_path,
        quantized_wav=args.quantized,
        save_audio=args.save_audio,
        debug=args.debug,
        metada_name="metada.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file for training.", required=True)
    parser.add_argument("--checkpoint_path", type=str, help="Model file to be restored.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to save mel specs", required=True)
    parser.add_argument("--debug", default=False, action="store_true", help="Save audio files for debug")
    parser.add_argument("--save_audio", default=False, action="store_true", help="Save audio files")
    parser.add_argument("--quantized", action="store_true", help="Save quantized audio files")
    args = parser.parse_args()

    c = load_config(args.config_path)
    main(args)
