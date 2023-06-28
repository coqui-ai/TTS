"""
Modified HuBERT model without kmeans.
Original author: https://github.com/lucidrains/
Modified by: https://www.github.com/gitmylo/
License: MIT
"""

# Modified code from https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/hubert_kmeans.py

import logging
from pathlib import Path

import fairseq
import torch
from einops import pack, unpack
from torch import nn
from torchaudio.functional import resample

logging.root.setLevel(logging.ERROR)


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    return t[..., seq_slice]


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class CustomHubert(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(self, checkpoint_path, target_sample_hz=16000, seq_len_multiple_of=None, output_layer=9, device=None):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        if device is not None:
            self.to(device)

        model_path = Path(checkpoint_path)

        assert model_path.exists(), f"path {checkpoint_path} does not exist"

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        if device is not None:
            model[0].to(device)

        self.model = model[0]
        self.model.eval()

    @property
    def groups(self):
        return 1

    @torch.no_grad()
    def forward(self, wav_input, flatten=True, input_sample_hz=None):
        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(
            wav_input,
            features_only=True,
            mask=False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
            output_layer=self.output_layer,
        )

        embed, packed_shape = pack([embed["x"]], "* d")

        # codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        codebook_indices = torch.from_numpy(embed.cpu().detach().numpy()).to(device)  # .long()

        if flatten:
            return codebook_indices

        (codebook_indices,) = unpack(codebook_indices, packed_shape, "*")
        return codebook_indices
