from typing import Dict

import numpy as np
import torch
from torch import nn


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
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
    style_text: str = None,
    d_vector: torch.Tensor = None,
    language_id: torch.Tensor = None,
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
            "style_text": style_text,
            "language_ids": language_id,
        },
    )
    return outputs


def trim_silence(wav, ap):
    return wav[: ap.find_endpoint(wav)]


def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav


def id_to_torch(aux_id, cuda=False):
    if aux_id is not None:
        aux_id = np.asarray(aux_id)
        aux_id = torch.from_numpy(aux_id)
    if cuda:
        return aux_id.cuda()
    return aux_id


def embedding_to_torch(d_vector, cuda=False):
    if d_vector is not None:
        d_vector = np.asarray(d_vector)
        d_vector = torch.from_numpy(d_vector).type(torch.FloatTensor)
        d_vector = d_vector.squeeze().unsqueeze(0)
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
    speaker_id=None,
    style_wav=None,
    style_text=None,
    use_griffin_lim=False,
    do_trim_silence=False,
    d_vector=None,
    language_id=None,
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

        speaker_id (int):
            Speaker ID passed to the speaker embedding layer in multi-speaker model. Defaults to None.

        style_wav (str | Dict[str, float]):
            Path or tensor to/of a waveform used for computing the style embedding based on GST or Capacitron.
            Defaults to None, meaning that Capacitron models will sample from the prior distribution to
            generate random but realistic prosody.

        style_text (str):
            Transcription of style_wav for Capacitron models. Defaults to None.

        enable_eos_bos_chars (bool):
            enable special chars for end of sentence and start of sentence. Defaults to False.

        do_trim_silence (bool):
            trim silence after synthesis. Defaults to False.

        d_vector (torch.Tensor):
            d-vector for multi-speaker models in share :math:`[1, D]`. Defaults to None.

        language_id (int):
            Language ID passed to the language embedding layer in multi-langual model. Defaults to None.
    """
    # GST or Capacitron processing
    # TODO: need to handle the case of setting both gst and capacitron to true somewhere
    style_mel = None
    if CONFIG.has("gst") and CONFIG.gst and style_wav is not None:
        if isinstance(style_wav, dict):
            style_mel = style_wav
        else:
            style_mel = compute_style_mel(style_wav, model.ap, cuda=use_cuda)

    if CONFIG.has("capacitron_vae") and CONFIG.use_capacitron_vae and style_wav is not None:
        style_mel = compute_style_mel(style_wav, model.ap, cuda=use_cuda)
        style_mel = style_mel.transpose(1, 2)  # [1, time, depth]

    language_name = None
    if language_id is not None:
        language = [k for k, v in model.language_manager.name_to_id.items() if v == language_id]
        assert len(language) == 1, "language_id must be a valid language"
        language_name = language[0]

    # convert text to sequence of token IDs
    text_inputs = np.asarray(
        model.tokenizer.text_to_ids(text, language=language_name),
        dtype=np.int32,
    )
    # pass tensors to backend
    if speaker_id is not None:
        speaker_id = id_to_torch(speaker_id, cuda=use_cuda)

    if d_vector is not None:
        d_vector = embedding_to_torch(d_vector, cuda=use_cuda)

    if language_id is not None:
        language_id = id_to_torch(language_id, cuda=use_cuda)

    if not isinstance(style_mel, dict):
        # GST or Capacitron style mel
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        if style_text is not None:
            style_text = np.asarray(
                model.tokenizer.text_to_ids(style_text, language=language_id),
                dtype=np.int32,
            )
            style_text = numpy_to_torch(style_text, torch.long, cuda=use_cuda)
            style_text = style_text.unsqueeze(0)

    text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=use_cuda)
    text_inputs = text_inputs.unsqueeze(0)
    # synthesize voice
    outputs = run_model_torch(
        model,
        text_inputs,
        speaker_id,
        style_mel,
        style_text,
        d_vector=d_vector,
        language_id=language_id,
    )
    model_outputs = outputs["model_outputs"]
    model_outputs = model_outputs[0].data.cpu().numpy()
    alignments = outputs["alignments"]

    # convert outputs to numpy
    # plot results
    wav = None
    model_outputs = model_outputs.squeeze()
    if model_outputs.ndim == 2:  # [T, C_spec]
        if use_griffin_lim:
            wav = inv_spectrogram(model_outputs, model.ap, CONFIG)
            # trim silence
            if do_trim_silence:
                wav = trim_silence(wav, model.ap)
    else:  # [T,]
        wav = model_outputs
    return_dict = {
        "wav": wav,
        "alignments": alignments,
        "text_inputs": text_inputs,
        "outputs": outputs,
    }
    return return_dict


def transfer_voice(
    model,
    CONFIG,
    use_cuda,
    reference_wav,
    speaker_id=None,
    d_vector=None,
    reference_speaker_id=None,
    reference_d_vector=None,
    do_trim_silence=False,
    use_griffin_lim=False,
):
    """Synthesize voice for the given text using Griffin-Lim vocoder or just compute output features to be passed to
    the vocoder model.

    Args:
        model (TTS.tts.models):
            The TTS model to synthesize audio with.

        CONFIG (Coqpit):
            Model configuration.

        use_cuda (bool):
            Enable/disable CUDA.

        reference_wav (str):
            Path of reference_wav to be used to voice conversion.

        speaker_id (int):
            Speaker ID passed to the speaker embedding layer in multi-speaker model. Defaults to None.

        d_vector (torch.Tensor):
            d-vector for multi-speaker models in share :math:`[1, D]`. Defaults to None.

        reference_speaker_id (int):
            Reference Speaker ID passed to the speaker embedding layer in multi-speaker model. Defaults to None.

        reference_d_vector (torch.Tensor):
            Reference d-vector for multi-speaker models in share :math:`[1, D]`. Defaults to None.

        enable_eos_bos_chars (bool):
            enable special chars for end of sentence and start of sentence. Defaults to False.

        do_trim_silence (bool):
            trim silence after synthesis. Defaults to False.
    """
    # pass tensors to backend
    if speaker_id is not None:
        speaker_id = id_to_torch(speaker_id, cuda=use_cuda)

    if d_vector is not None:
        d_vector = embedding_to_torch(d_vector, cuda=use_cuda)

    if reference_d_vector is not None:
        reference_d_vector = embedding_to_torch(reference_d_vector, cuda=use_cuda)

    # load reference_wav audio
    reference_wav = embedding_to_torch(
        model.ap.load_wav(
            reference_wav, sr=model.args.encoder_sample_rate if model.args.encoder_sample_rate else model.ap.sample_rate
        ),
        cuda=use_cuda,
    )

    if hasattr(model, "module"):
        _func = model.module.inference_voice_conversion
    else:
        _func = model.inference_voice_conversion
    model_outputs = _func(reference_wav, speaker_id, d_vector, reference_speaker_id, reference_d_vector)

    # convert outputs to numpy
    # plot results
    wav = None
    model_outputs = model_outputs.squeeze()
    if model_outputs.ndim == 2:  # [T, C_spec]
        if use_griffin_lim:
            wav = inv_spectrogram(model_outputs, model.ap, CONFIG)
            # trim silence
            if do_trim_silence:
                wav = trim_silence(wav, model.ap)
    else:  # [T,]
        wav = model_outputs

    return wav
