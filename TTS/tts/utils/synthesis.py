import os
from typing import Dict

import numpy as np
import pkg_resources
import torch
from torch import nn

from .text import phoneme_to_sequence, text_to_sequence

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

installed = {pkg.key for pkg in pkg_resources.working_set}  # pylint: disable=not-an-iterable
if "tensorflow" in installed or "tensorflow-gpu" in installed:
    import tensorflow as tf


def text_to_seq(text, CONFIG, custom_symbols=None):
    text_cleaner = [CONFIG.text_cleaner]
    # text ot phonemes to sequence vector
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(
                text,
                text_cleaner,
                CONFIG.phoneme_language,
                CONFIG.enable_eos_bos_chars,
                tp=CONFIG.characters,
                add_blank=CONFIG.add_blank,
                use_espeak_phonemes=CONFIG.use_espeak_phonemes,
                custom_symbols=custom_symbols,
            ),
            dtype=np.int32,
        )
    else:
        seq = np.asarray(
            text_to_sequence(
                text, text_cleaner, tp=CONFIG.characters, add_blank=CONFIG.add_blank, custom_symbols=custom_symbols
            ),
            dtype=np.int32,
        )
    return seq


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor


def numpy_to_tf(np_array, dtype):
    if np_array is None:
        return None
    tensor = tf.convert_to_tensor(np_array, dtype=dtype)
    return tensor


def compute_style_mel(style_wav, ap, cuda=False):
    style_mel = torch.FloatTensor(ap.melspectrogram(ap.load_wav(style_wav, sr=ap.sample_rate))).unsqueeze(0)
    if cuda:
        return style_mel.cuda()
    return style_mel


def run_model_torch(
    model: nn.Module,
    inputs: torch.Tensor,
    speaker_id: int = None,
    style_mel: torch.Tensor = None,
    d_vector: torch.Tensor = None,
) -> Dict:
    """Run a torch model for inference. It does not support batch inference.

    Args:
        model (nn.Module): The model to run inference.
        inputs (torch.Tensor): Input tensor with character ids.
        speaker_id (int, optional): Input speaker ids for multi-speaker models. Defaults to None.
        style_mel (torch.Tensor, optional): Spectrograms used for voice styling . Defaults to None.
        d_vector (torch.Tensor, optional): d-vector for multi-speaker models    . Defaults to None.

    Returns:
        Dict: model outputs.
    """
    input_lengths = torch.tensor(inputs.shape[1:2]).to(inputs.device)
    if hasattr(model, "module"):
        _func = model.module.inference
    else:
        _func = model.inference
    outputs = _func(
        inputs,
        aux_input={
            "x_lengths": input_lengths,
            "speaker_ids": speaker_id,
            "d_vectors": d_vector,
            "style_mel": style_mel,
        },
    )
    return outputs


def run_model_tf(model, inputs, CONFIG, speaker_id=None, style_mel=None):
    if CONFIG.gst and style_mel is not None:
        raise NotImplementedError(" [!] GST inference not implemented for TF")
    if speaker_id is not None:
        raise NotImplementedError(" [!] Multi-Speaker not implemented for TF")
    # TODO: handle multispeaker case
    decoder_output, postnet_output, alignments, stop_tokens = model(inputs, training=False)
    return decoder_output, postnet_output, alignments, stop_tokens


def run_model_tflite(model, inputs, CONFIG, speaker_id=None, style_mel=None):
    if CONFIG.gst and style_mel is not None:
        raise NotImplementedError(" [!] GST inference not implemented for TfLite")
    if speaker_id is not None:
        raise NotImplementedError(" [!] Multi-Speaker not implemented for TfLite")
    # get input and output details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    # reshape input tensor for the new input shape
    model.resize_tensor_input(input_details[0]["index"], inputs.shape)
    model.allocate_tensors()
    detail = input_details[0]
    # input_shape = detail['shape']
    model.set_tensor(detail["index"], inputs)
    # run the model
    model.invoke()
    # collect outputs
    decoder_output = model.get_tensor(output_details[0]["index"])
    postnet_output = model.get_tensor(output_details[1]["index"])
    # tflite model only returns feature frames
    return decoder_output, postnet_output, None, None


def parse_outputs_tf(postnet_output, decoder_output, alignments, stop_tokens):
    postnet_output = postnet_output[0].numpy()
    decoder_output = decoder_output[0].numpy()
    alignment = alignments[0].numpy()
    stop_tokens = stop_tokens[0].numpy()
    return postnet_output, decoder_output, alignment, stop_tokens


def parse_outputs_tflite(postnet_output, decoder_output):
    postnet_output = postnet_output[0]
    decoder_output = decoder_output[0]
    return postnet_output, decoder_output


def trim_silence(wav, ap):
    return wav[: ap.find_endpoint(wav)]


def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav


def speaker_id_to_torch(speaker_id, cuda=False):
    if speaker_id is not None:
        speaker_id = np.asarray(speaker_id)
        speaker_id = torch.from_numpy(speaker_id)
    if cuda:
        return speaker_id.cuda()
    return speaker_id


def embedding_to_torch(d_vector, cuda=False):
    if d_vector is not None:
        d_vector = np.asarray(d_vector)
        d_vector = torch.from_numpy(d_vector).type(torch.FloatTensor)
    if cuda:
        return d_vector.cuda()
    return d_vector


# TODO: perform GL with pytorch for batching
def apply_griffin_lim(inputs, input_lens, CONFIG, ap):
    """Apply griffin-lim to each sample iterating throught the first dimension.
    Args:
        inputs (Tensor or np.Array): Features to be converted by GL. First dimension is the batch size.
        input_lens (Tensor or np.Array): 1D array of sample lengths.
        CONFIG (Dict): TTS config.
        ap (AudioProcessor): TTS audio processor.
    """
    wavs = []
    for idx, spec in enumerate(inputs):
        wav_len = (input_lens[idx] * ap.hop_length) - ap.hop_length  # inverse librosa padding
        wav = inv_spectrogram(spec, ap, CONFIG)
        # assert len(wav) == wav_len, f" [!] wav lenght: {len(wav)} vs expected: {wav_len}"
        wavs.append(wav[:wav_len])
    return wavs


def synthesis(
    model,
    text,
    CONFIG,
    use_cuda,
    ap,
    speaker_id=None,
    style_wav=None,
    enable_eos_bos_chars=False,  # pylint: disable=unused-argument
    use_griffin_lim=False,
    do_trim_silence=False,
    d_vector=None,
    backend="torch",
):
    """Synthesize voice for the given text using Griffin-Lim vocoder or just compute output features to be passed to
    the vocoder model.

    Args:
        model (TTS.tts.models):
            The TTS model to synthesize audio with.

        text (str):
            The input text to convert to speech.

        CONFIG (Coqpit):
            Model configuration.

        use_cuda (bool):
            Enable/disable CUDA.

        ap (TTS.tts.utils.audio.AudioProcessor):
            The audio processor for extracting features and pre/post-processing audio.

        speaker_id (int):
            Speaker ID passed to the speaker embedding layer in multi-speaker model. Defaults to None.

        style_wav (str | Dict[str, float]):
            Path or tensor to/of a waveform used for computing the style embedding. Defaults to None.

        enable_eos_bos_chars (bool):
            enable special chars for end of sentence and start of sentence. Defaults to False.

        do_trim_silence (bool):
            trim silence after synthesis. Defaults to False.

        d_vector (torch.Tensor):
            d-vector for multi-speaker models in share :math:`[1, D]`. Defaults to None.

        backend (str):
            tf or torch. Defaults to "torch".
    """
    # GST processing
    style_mel = None
    custom_symbols = None
    if style_wav:
        style_mel = compute_style_mel(style_wav, ap, cuda=use_cuda)
    elif CONFIG.has("gst") and CONFIG.gst and not style_wav:
        if CONFIG.gst.gst_style_input_weights:
            style_mel = CONFIG.gst.gst_style_input_weights
    if hasattr(model, "make_symbols"):
        custom_symbols = model.make_symbols(CONFIG)
    # preprocess the given text
    text_inputs = text_to_seq(text, CONFIG, custom_symbols=custom_symbols)
    # pass tensors to backend
    if backend == "torch":
        if speaker_id is not None:
            speaker_id = speaker_id_to_torch(speaker_id, cuda=use_cuda)

        if d_vector is not None:
            d_vector = embedding_to_torch(d_vector, cuda=use_cuda)

        if not isinstance(style_mel, dict):
            style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=use_cuda)
        text_inputs = text_inputs.unsqueeze(0)
    elif backend in ["tf", "tflite"]:
        # TODO: handle speaker id for tf model
        style_mel = numpy_to_tf(style_mel, tf.float32)
        text_inputs = numpy_to_tf(text_inputs, tf.int32)
        text_inputs = tf.expand_dims(text_inputs, 0)
    # synthesize voice
    if backend == "torch":
        outputs = run_model_torch(model, text_inputs, speaker_id, style_mel, d_vector=d_vector)
        model_outputs = outputs["model_outputs"]
        model_outputs = model_outputs[0].data.cpu().numpy()
        alignments = outputs["alignments"]
    elif backend == "tf":
        decoder_output, postnet_output, alignments, stop_tokens = run_model_tf(
            model, text_inputs, CONFIG, speaker_id, style_mel
        )
        model_outputs, decoder_output, alignments, stop_tokens = parse_outputs_tf(
            postnet_output, decoder_output, alignments, stop_tokens
        )
    elif backend == "tflite":
        decoder_output, postnet_output, alignments, stop_tokens = run_model_tflite(
            model, text_inputs, CONFIG, speaker_id, style_mel
        )
        model_outputs, decoder_output = parse_outputs_tflite(postnet_output, decoder_output)
    # convert outputs to numpy
    # plot results
    wav = None
    if hasattr(model, "END2END") and model.END2END:
        wav = model_outputs.squeeze(0)
    else:
        if use_griffin_lim:
            wav = inv_spectrogram(model_outputs, ap, CONFIG)
            # trim silence
            if do_trim_silence:
                wav = trim_silence(wav, ap)
    return_dict = {
        "wav": wav,
        "alignments": alignments,
        "text_inputs": text_inputs,
        "outputs": outputs,
    }
    return return_dict
