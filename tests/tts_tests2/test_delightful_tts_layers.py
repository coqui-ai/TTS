import torch

from TTS.tts.configs.delightful_tts_config import DelightfulTTSConfig
from TTS.tts.layers.delightful_tts.acoustic_model import AcousticModel
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, VocoderConfig
from TTS.tts.utils.helpers import rand_segments
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.vocoder.models.hifigan_generator import HifiganGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = DelightfulTtsArgs()
v_args = VocoderConfig()


config = DelightfulTTSConfig(
    model_args=args,
    # compute_f0=True,
    # f0_cache_path=os.path.join(output_path, "f0_cache"),
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    # phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
)

tokenizer, config = TTSTokenizer.init_from_config(config)


def test_acoustic_model():
    dummy_tokens = torch.rand((1, 41)).long().to(device)
    dummy_text_lens = torch.tensor([41]).long().to(device)
    dummy_spec = torch.rand((1, 100, 207)).to(device)
    dummy_spec_lens = torch.tensor([207]).to(device)
    dummy_pitch = torch.rand((1, 1, 207)).long().to(device)
    dummy_energy = torch.rand((1, 1, 207)).long().to(device)

    args.out_channels = 100
    args.num_mels = 100

    acoustic_model = AcousticModel(args=args, tokenizer=tokenizer, speaker_manager=None).to(device)
    acoustic_model = acoustic_model.train()

    output = acoustic_model(
        tokens=dummy_tokens,
        src_lens=dummy_text_lens,
        mel_lens=dummy_spec_lens,
        mels=dummy_spec,
        pitches=dummy_pitch,
        energies=dummy_energy,
        attn_priors=None,
        d_vectors=None,
        speaker_idx=None,
    )
    assert list(output["model_outputs"].shape) == [1, 207, 100]
    # output["model_outputs"].sum().backward()


def test_hifi_decoder():
    dummy_input = torch.rand((1, 207, 100)).to(device)
    dummy_spec_lens = torch.tensor([207]).to(device)

    waveform_decoder = HifiganGenerator(
        100,
        1,
        v_args.resblock_type_decoder,
        v_args.resblock_dilation_sizes_decoder,
        v_args.resblock_kernel_sizes_decoder,
        v_args.upsample_kernel_sizes_decoder,
        v_args.upsample_initial_channel_decoder,
        v_args.upsample_rates_decoder,
        inference_padding=0,
        cond_channels=0,
        conv_pre_weight_norm=False,
        conv_post_weight_norm=False,
        conv_post_bias=False,
    ).to(device)
    waveform_decoder = waveform_decoder.train()

    vocoder_input_slices, slice_ids = rand_segments(  # pylint: disable=unused-variable
        x=dummy_input.transpose(1, 2),
        x_lengths=dummy_spec_lens,
        segment_size=32,
        let_short_samples=True,
        pad_short=True,
    )

    outputs = waveform_decoder(x=vocoder_input_slices.detach())
    assert list(outputs.shape) == [1, 1, 8192]
    # outputs.sum().backward()
