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
from TTS.utils.audio.numpy_transforms import quantize
from TTS.utils.generic_utils import count_parameters

use_cuda = torch.cuda.is_available()


def setup_loader(ap, r, verbose=False):
    tokenizer, _ = TTSTokenizer.init_from_config(c)
    dataset = TTSDataset(
        outputs_per_step=r,
        compute_linear_spec=False,
        samples=meta_data,
        tokenizer=tokenizer,
        ap=ap,
        batch_group_size=0,
        min_text_len=c.min_text_len,
        max_text_len=c.max_text_len,
        min_audio_len=c.min_audio_len,
        max_audio_len=c.max_audio_len,
        phoneme_cache_path=c.phoneme_cache_path,
        precompute_num_workers=0,
        use_noise_augment=False,
        verbose=verbose,
        speaker_id_mapping=speaker_manager.name_to_id if c.use_speaker_embedding else None,
        d_vector_mapping=speaker_manager.embeddings if c.use_d_vector_file else None,
    )

    if c.use_phonemes and c.compute_input_seq_cache:
        # precompute phonemes to have a better estimate of sequence lengths.
        dataset.compute_input_seq(c.num_loader_workers)
    dataset.preprocess_samples()

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
    text_input = data["token_id"]
    text_lengths = data["token_id_lengths"]
    mel_input = data["mel"]
    mel_lengths = data["mel_lengths"]
    item_idx = data["item_idxs"]
    d_vectors = data["d_vectors"]
    speaker_ids = data["speaker_ids"]
    attn_mask = data["attns"]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if d_vectors is not None:
            d_vectors = d_vectors.cuda(non_blocking=True)
        if attn_mask is not None:
            attn_mask = attn_mask.cuda(non_blocking=True)
    return (
        text_input,
        text_lengths,
        mel_input,
        mel_lengths,
        speaker_ids,
        d_vectors,
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
    speaker_ids=None,
    d_vectors=None,
):
    if model_name == "glow_tts":
        speaker_c = None
        if speaker_ids is not None:
            speaker_c = speaker_ids
        elif d_vectors is not None:
            speaker_c = d_vectors
        outputs = model.inference_with_MAS(
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            aux_input={"d_vectors": speaker_c, "speaker_ids": speaker_ids},
        )
        model_output = outputs["model_outputs"]
        model_output = model_output.detach().cpu().numpy()

    elif "tacotron" in model_name:
        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = model(text_input, text_lengths, mel_input, mel_lengths, aux_input)
        postnet_outputs = outputs["model_outputs"]
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
    data_loader, model, ap, output_path, quantize_bits=0, save_audio=False, debug=False, metada_name="metada.txt"
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
            d_vectors,
            _,
            _,
            _,
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
            speaker_ids,
            d_vectors,
        )

        for idx in range(text_input.shape[0]):
            wav_file_path = item_idx[idx]
            wav = ap.load_wav(wav_file_path)
            _, wavq_path, mel_path, wav_gl_path, wav_path = set_filename(wav_file_path, output_path)

            # quantize and save wav
            if quantize_bits > 0:
                wavq = quantize(wav, quantize_bits)
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

    with open(os.path.join(output_path, metada_name), "w", encoding="utf-8") as f:
        for data in export_metadata:
            f.write(f"{data[0]}|{data[1]+'.npy'}\n")


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

    # init speaker manager
    if c.use_speaker_embedding:
        speaker_manager = SpeakerManager(data_items=meta_data)
    elif c.use_d_vector_file:
        speaker_manager = SpeakerManager(d_vectors_file_path=c.d_vector_file)
    else:
        speaker_manager = None

    # setup model
    model = setup_model(c)

    # restore model
    model.load_checkpoint(c, args.checkpoint_path, eval=True)

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
        quantize_bits=args.quantize_bits,
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
    parser.add_argument("--quantize_bits", type=int, default=0, help="Save quantized audio files if non-zero")
    parser.add_argument("--eval", type=bool, help="compute eval.", default=True)
    args = parser.parse_args()

    c = load_config(args.config_path)
    c.audio.trim_silence = False
    main(args)
