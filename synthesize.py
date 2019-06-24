import os
import time
import argparse
import torch
import string

from utils.synthesis import synthesis
from utils.generic_utils import load_config, setup_model
from utils.text.symbols import symbols, phonemes
from utils.audio import AudioProcessor

from WaveRNN.models.wavernn import Model as VocoderModel


def tts(model,
        vocoder_model,
        C,
        VC,
        text,
        ap,
        use_cuda,
        batched_vocoder,
        figures=False):
    t_1 = time.time()
    use_vocoder_model = vocoder_model is not None
    waveform, alignment, decoder_outputs, postnet_output, stop_tokens = synthesis(
        model, text, C, use_cuda, ap, False, C.enable_eos_bos_chars)
    if C.model == "Tacotron" and use_vocoder_model:
        postnet_output = ap.out_linear_to_mel(postnet_output.T).T
    if use_vocoder_model:
        vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
        waveform = vocoder_model.generate(
            vocoder_input.cuda() if use_cuda else vocoder_input,
            batched=batched_vocoder,
            target=11000,
            overlap=550)
    print(" >  Run-time: {}".format(time.time() - t_1))
    return alignment, postnet_output, stop_tokens, waveform


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'text', type=str, help='Text to generate speech.')
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to model config file.'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to model file.',
    )
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save final wav file.',
    )
    parser.add_argument(
        '--use_cuda', type=bool, help='Run model on CUDA.', default=False)
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help=
        'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default="",
    )
    parser.add_argument(
        '--vocoder_config_path',
        type=str,
        help='Path to vocoder model config file.',
        default="")
    parser.add_argument(
        '--batched_vocoder',
        type=bool,
        help="If True, vocoder model uses faster batch processing.",
        default=True)
    args = parser.parse_args()

    if args.vocoder_path != "":
        assert args.use_cuda, " [!] Enable cuda for vocoder."
    # load the config
    C = load_config(args.config_path)
    C.forward_attn_mask = True

    # load the audio processor
    ap = AudioProcessor(**C.audio)

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, C)
    cp = torch.load(args.model_path)
    model.load_state_dict(cp['model'])
    model.eval()
    if args.use_cuda:
        model.cuda()

    # load vocoder model
    if args.vocoder_path != "":
        VC = load_config(args.vocoder_config_path)
        bits = 10
        vocoder_model = VocoderModel(
            rnn_dims=512,
            fc_dims=512,
            mode=VC.mode,
            mulaw=VC.mulaw,
            pad=VC.pad,
            upsample_factors=VC.upsample_factors,
            feat_dims=VC.audio["num_mels"],
            compute_dims=128,
            res_out_dims=128,
            res_blocks=10,
            hop_length=ap.hop_length,
            sample_rate=ap.sample_rate,
        )

        check = torch.load(args.vocoder_path)
        vocoder_model.load_state_dict(check['model'])
        vocoder_model.eval()
        if args.use_cuda:
            vocoder_model.cuda()
    else:
        vocoder_model = None
        VC = None

    # synthesize voice
    print(" > Text: {}".format(args.text))
    _, _, _, wav = tts(
        model,
        vocoder_model,
        C,
        VC,
        args.text,
        ap,
        args.use_cuda,
        args.batched_vocoder,
        figures=False)

    # save the results
    file_name = args.text.replace(" ", "_")
    file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', '')))+'.wav'
    out_path = os.path.join(args.out_path, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
