import pkg_resources
installed = {pkg.key for pkg in pkg_resources.working_set}  #pylint: disable=not-an-iterable
if 'tensorflow' in installed or 'tensorflow-gpu' in installed:
    import tensorflow as tf
import torch
import numpy as np
from .text import text_to_sequence, phoneme_to_sequence


def text_to_seqvec(text, CONFIG):
    text_cleaner = [CONFIG.text_cleaner]
    # text ot phonemes to sequence vector
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(text, text_cleaner, CONFIG.phoneme_language,
                                CONFIG.enable_eos_bos_chars,
                                tp=CONFIG.characters if 'characters' in CONFIG.keys() else None,
                                add_blank=CONFIG['add_blank'] if 'add_blank' in CONFIG.keys() else False),
            dtype=np.int32)
    else:
        seq = np.asarray(
            text_to_sequence(text, text_cleaner, tp=CONFIG.characters if 'characters' in CONFIG.keys() else None,
            add_blank=CONFIG['add_blank'] if 'add_blank' in CONFIG.keys() else False), dtype=np.int32)
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
    style_mel = torch.FloatTensor(ap.melspectrogram(
        ap.load_wav(style_wav, sr=ap.sample_rate))).unsqueeze(0)
    if cuda:
        return style_mel.cuda()
    return style_mel


def run_model_torch(model, inputs, CONFIG, truncated, speaker_id=None, style_mel=None, speaker_embeddings=None):
    if 'tacotron' in CONFIG.model.lower():
        if CONFIG.use_gst:
            decoder_output, postnet_output, alignments, stop_tokens = model.inference(
                inputs, style_mel=style_mel, speaker_ids=speaker_id, speaker_embeddings=speaker_embeddings)
        else:
            if truncated:
                decoder_output, postnet_output, alignments, stop_tokens = model.inference_truncated(
                    inputs, speaker_ids=speaker_id, speaker_embeddings=speaker_embeddings)
            else:
                decoder_output, postnet_output, alignments, stop_tokens = model.inference(
                    inputs, speaker_ids=speaker_id, speaker_embeddings=speaker_embeddings)
    elif 'glow' in CONFIG.model.lower():
        inputs_lengths = torch.tensor(inputs.shape[1:2]).to(inputs.device)  # pylint: disable=not-callable
        if hasattr(model, 'module'):
            # distributed model
            postnet_output, _, _, _, alignments, _, _ = model.module.inference(inputs, inputs_lengths, g=speaker_id if speaker_id is not None else speaker_embeddings)
        else:
            postnet_output, _, _, _, alignments, _, _ = model.inference(inputs, inputs_lengths, g=speaker_id if speaker_id is not None else speaker_embeddings)
        postnet_output = postnet_output.permute(0, 2, 1)
        # these only belong to tacotron models.
        decoder_output = None
        stop_tokens = None
    elif 'speedy_speech' in CONFIG.model.lower():
        inputs_lengths = torch.tensor(inputs.shape[1:2]).to(inputs.device)  # pylint: disable=not-callable
        if hasattr(model, 'module'):
            # distributed model
            postnet_output, alignments= model.module.inference(inputs, inputs_lengths, g=speaker_id if speaker_id is not None else speaker_embeddings)
        else:
            postnet_output, alignments= model.inference(inputs, inputs_lengths, g=speaker_id if speaker_id is not None else speaker_embeddings)
        postnet_output = postnet_output.permute(0, 2, 1)
        # these only belong to tacotron models.
        decoder_output = None
        stop_tokens = None
    return decoder_output, postnet_output, alignments, stop_tokens


def run_model_tf(model, inputs, CONFIG, truncated, speaker_id=None, style_mel=None):
    if CONFIG.use_gst and style_mel is not None:
        raise NotImplementedError(' [!] GST inference not implemented for TF')
    if truncated:
        raise NotImplementedError(' [!] Truncated inference not implemented for TF')
    if speaker_id is not None:
        raise NotImplementedError(' [!] Multi-Speaker not implemented for TF')
    # TODO: handle multispeaker case
    decoder_output, postnet_output, alignments, stop_tokens = model(
        inputs, training=False)
    return decoder_output, postnet_output, alignments, stop_tokens


def run_model_tflite(model, inputs, CONFIG, truncated, speaker_id=None, style_mel=None):
    if CONFIG.use_gst and style_mel is not None:
        raise NotImplementedError(' [!] GST inference not implemented for TfLite')
    if truncated:
        raise NotImplementedError(' [!] Truncated inference not implemented for TfLite')
    if speaker_id is not None:
        raise NotImplementedError(' [!] Multi-Speaker not implemented for TfLite')
    # get input and output details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    # reshape input tensor for the new input shape
    model.resize_tensor_input(input_details[0]['index'], inputs.shape)
    model.allocate_tensors()
    detail = input_details[0]
    # input_shape = detail['shape']
    model.set_tensor(detail['index'], inputs)
    # run the model
    model.invoke()
    # collect outputs
    decoder_output = model.get_tensor(output_details[0]['index'])
    postnet_output = model.get_tensor(output_details[1]['index'])
    # tflite model only returns feature frames
    return decoder_output, postnet_output, None, None


def parse_outputs_torch(postnet_output, decoder_output, alignments, stop_tokens):
    postnet_output = postnet_output[0].data.cpu().numpy()
    decoder_output = None if decoder_output is None else decoder_output[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    stop_tokens = None if stop_tokens is None else stop_tokens[0].cpu().numpy()
    return postnet_output, decoder_output, alignment, stop_tokens


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
    return wav[:ap.find_endpoint(wav)]


def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav


def id_to_torch(speaker_id, cuda=False):
    if speaker_id is not None:
        speaker_id = np.asarray(speaker_id)
        # TODO: test this for tacotron models
        speaker_id = torch.from_numpy(speaker_id)
    if cuda:
        return speaker_id.cuda()
    return speaker_id


def embedding_to_torch(speaker_embedding, cuda=False):
    if speaker_embedding is not None:
        speaker_embedding = np.asarray(speaker_embedding)
        speaker_embedding = torch.from_numpy(speaker_embedding).unsqueeze(0).type(torch.FloatTensor)
    if cuda:
        return speaker_embedding.cuda()
    return speaker_embedding


# TODO: perform GL with pytorch for batching
def apply_griffin_lim(inputs, input_lens, CONFIG, ap):
    '''Apply griffin-lim to each sample iterating throught the first dimension.
    Args:
        inputs (Tensor or np.Array): Features to be converted by GL. First dimension is the batch size.
        input_lens (Tensor or np.Array): 1D array of sample lengths.
        CONFIG (Dict): TTS config.
        ap (AudioProcessor): TTS audio processor.
    '''
    wavs = []
    for idx, spec in enumerate(inputs):
        wav_len = (input_lens[idx] * ap.hop_length) - ap.hop_length  # inverse librosa padding
        wav = inv_spectrogram(spec, ap, CONFIG)
        # assert len(wav) == wav_len, f" [!] wav lenght: {len(wav)} vs expected: {wav_len}"
        wavs.append(wav[:wav_len])
    return wavs


def synthesis(model,
              text,
              CONFIG,
              use_cuda,
              ap,
              speaker_id=None,
              style_wav=None,
              truncated=False,
              enable_eos_bos_chars=False, #pylint: disable=unused-argument
              use_griffin_lim=False,
              do_trim_silence=False,
              speaker_embedding=None,
              backend='torch'):
    """Synthesize voice for the given text.

        Args:
            model (TTS.tts.models): model to synthesize.
            text (str): target text
            CONFIG (dict): config dictionary to be loaded from config.json.
            use_cuda (bool): enable cuda.
            ap (TTS.tts.utils.audio.AudioProcessor): audio processor to process
                model outputs.
            speaker_id (int): id of speaker
            style_wav (str): Uses for style embedding of GST.
            truncated (bool): keep model states after inference. It can be used
                for continuous inference at long texts.
            enable_eos_bos_chars (bool): enable special chars for end of sentence and start of sentence.
            do_trim_silence (bool): trim silence after synthesis.
            backend (str): tf or torch
    """
    # GST processing
    style_mel = None
    if 'use_gst' in CONFIG.keys() and CONFIG.use_gst and style_wav is not None:
        if isinstance(style_wav, dict):
            style_mel = style_wav
        else:
            style_mel = compute_style_mel(style_wav, ap, cuda=use_cuda)
    # preprocess the given text
    inputs = text_to_seqvec(text, CONFIG)
    # pass tensors to backend
    if backend == 'torch':
        if speaker_id is not None:
            speaker_id = id_to_torch(speaker_id, cuda=use_cuda)

        if speaker_embedding is not None:
            speaker_embedding = embedding_to_torch(speaker_embedding, cuda=use_cuda)

        if not isinstance(style_mel, dict):
            style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        inputs = numpy_to_torch(inputs, torch.long, cuda=use_cuda)
        inputs = inputs.unsqueeze(0)
    elif backend == 'tf':
        # TODO: handle speaker id for tf model
        style_mel = numpy_to_tf(style_mel, tf.float32)
        inputs = numpy_to_tf(inputs, tf.int32)
        inputs = tf.expand_dims(inputs, 0)
    elif backend == 'tflite':
        style_mel = numpy_to_tf(style_mel, tf.float32)
        inputs = numpy_to_tf(inputs, tf.int32)
        inputs = tf.expand_dims(inputs, 0)
    # synthesize voice
    if backend == 'torch':
        decoder_output, postnet_output, alignments, stop_tokens = run_model_torch(
            model, inputs, CONFIG, truncated, speaker_id, style_mel, speaker_embeddings=speaker_embedding)
        postnet_output, decoder_output, alignment, stop_tokens = parse_outputs_torch(
            postnet_output, decoder_output, alignments, stop_tokens)
    elif backend == 'tf':
        decoder_output, postnet_output, alignments, stop_tokens = run_model_tf(
            model, inputs, CONFIG, truncated, speaker_id, style_mel)
        postnet_output, decoder_output, alignment, stop_tokens = parse_outputs_tf(
            postnet_output, decoder_output, alignments, stop_tokens)
    elif backend == 'tflite':
        decoder_output, postnet_output, alignment, stop_tokens = run_model_tflite(
            model, inputs, CONFIG, truncated, speaker_id, style_mel)
        postnet_output, decoder_output = parse_outputs_tflite(
            postnet_output, decoder_output)
    # convert outputs to numpy
    # plot results
    wav = None
    if use_griffin_lim:
        wav = inv_spectrogram(postnet_output, ap, CONFIG)
        # trim silence
        if do_trim_silence:
            wav = trim_silence(wav, ap)
    return wav, alignment, decoder_output, postnet_output, stop_tokens, inputs
