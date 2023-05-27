from functools import reduce
from einops import rearrange, pack, unpack

import torch
from torch import nn

from vector_quantize_pytorch import ResidualVQ

from encodec import EncodecModel
from encodec.utils import _linear_overlap_add

class EncodecWrapper(nn.Module):
    """
    Support pretrained 24kHz Encodec by Meta AI, if you want to skip training SoundStream.

    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    - see if we can use the 48kHz model, which is specifically for music. Right now we're using the 24kHz model because
        that's what was used in MusicLM and avoids any resampling issues.
    -

    """
    def __init__(
        self,
        target_sample_hz = 24000,
        strides = (2, 4, 5, 8),
        num_quantizers = 8,
    ):
        super().__init__()
        # Instantiate a pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False # this means we don't need to scale codes e.g. when running model.encode(wav)

        # bandwidth affects num quantizers used: https://github.com/facebookresearch/encodec/pull/41
        self.model.set_target_bandwidth(6.0)
        assert num_quantizers == 8, "assuming 8 quantizers for now, see bandwidth comment above"

        # Fields that SoundStream has that get used externally. We replicate them here.
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"

        self.codebook_dim = 128
        self.rq_groups = 1
        self.num_quantizers = num_quantizers
        self.strides = strides # used in seq_len_multiple_of

        # cross entropy loss to indices passed in on l2 distance logits introduced in vector-quantize-pytorch 1.2.2

        self.rq = ResidualVQ(
            dim = 128,
            codebook_size = 1024,
            num_quantizers = 8
        )

        # copy codebook over to ResidualVQ for cross entropy loss logic from naturalspeech2
        # luckily, it seems Meta AI basically used my ResidualVQ code verbatim. makes porting it over easy

        for encodec_rq_layer, rq_layer in zip(self.model.quantizer.vq.layers, self.rq.layers):
            encodec_codebook = dict(encodec_rq_layer._codebook.named_buffers()).get('embed')
            vq_codebook = dict(rq_layer._codebook.named_buffers()).get('embed')

            encodec_codebook = rearrange(encodec_codebook, '... -> 1 ...')
            vq_codebook.copy_(encodec_codebook)

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        return_encoded = False,
        **kwargs
    ):

        x, ps = pack([x], '* n')

        # kwargs for stuff like return_encoded=True, which SoundStream uses but Encodec doesn't
        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."
        # Unlike in the Encodec sample code in its README, x has already been resampled so we don't need to call
        # convert_audio and unsqueeze. The convert_audio function also doesn't play nicely with batches.

        # b = batch, t = timesteps, 1 channel for the 24kHz model, 2 channels for the 48kHz model
        wav = rearrange(x, f'b t -> b {self.model.channels} t')

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
        # encoded_frames is a list of (frame, scale) tuples. Scale is a scalar but we don't use it. Frame is a tensor
        # of shape [batch, num_quantizers, num_samples_per_frame]. We want to concatenate the frames to get all the
        # timesteps concatenated.
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=1)  # [batch, num_quantizers, timesteps]
        # transformer code that uses codec expects codes to be [batch, timesteps, num_quantizers]
        codes = rearrange(codes, 'b q n -> b n q')  # result: [batch, timesteps, num_quantizers]
        # in original soundstream, is x, indices, commit_loss. But we only use indices in eval mode, so just keep that.

        # allow for returning of sum of quantized embeddings

        emb = None
        if return_encoded:
            emb = self.get_emb_from_indices(codes)

        emb, = unpack(emb, ps, '* n c')
        codes, = unpack(codes, ps, '* n q')

        return emb, codes, None

    def decode_from_codebook_indices(self, quantized_indices):
        # Input: batch x num tokens x num quantizers
        # Output: batch x 1 x num samples

        assert self.model.sample_rate == 24000,\
            "if changing to 48kHz, that model segments its audio into lengths of 1.0 second with 1% overlap, whereas " \
            "the 24kHz doesn't segment at all. this means the frame decode logic might change; this is a reminder to " \
            "double check that."
        # Since 24kHz pretrained doesn't do any segmenting, we have all the frames already (1 frame = 1 token in quantized_indices)

        # The following code is hacked in from self.model.decode() (Encodec version 0.1.1) where we skip the part about
        # scaling.
        # Shape: 1 x (num_frames * stride product). 1 because we have 1 frame (because no segmenting)
        frames = self._decode_frame(quantized_indices)
        result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        # TODO: I'm not overly pleased with this because when this function gets called, we just rearrange the result
        #   back to b n anyways, but we'll keep this as a temporary hack just to make things work for now
        return rearrange(result, 'b n -> b 1 n')

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')

    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)

    def _decode_frame(self, quantized_indices):
        # The following code is hacked in from self.model._decode_frame() (Encodec version 0.1.1) where we assume we've
        # already unwrapped the EncodedFrame
        # Input: batch x num tokens x num quantizers
        # Output: batch x new_num_samples, where new_num_samples is num_frames * stride product (may be slightly
        # larger than original num samples as a result, because the last frame might not be "fully filled" with samples
        # if num_samples doesn't divide perfectly).
        # num_frames == the number of acoustic tokens you have, one token per frame
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        # emb shape: batch x self.model.quantizer.dimension x T. Note self.model.quantizer.dimension is the embedding dimension
        return self.model.decoder(emb)