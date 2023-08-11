import logging
import os
import re
from glob import glob
from typing import Dict, List

import librosa
import numpy as np
import torch
import torchaudio
import tqdm
from encodec.utils import convert_audio
from scipy.special import softmax
from torch.nn import functional as F

from TTS.tts.layers.bark.hubert.hubert_manager import HubertManager
from TTS.tts.layers.bark.hubert.kmeans_hubert import CustomHubert
from TTS.tts.layers.bark.hubert.tokenizer import HubertTokenizer
from TTS.tts.layers.bark.load_model import clear_cuda_cache, inference_mode

logger = logging.getLogger(__name__)


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def get_voices(extra_voice_dirs: List[str] = []):  # pylint: disable=dangerous-default-value
    dirs = extra_voice_dirs
    voices: Dict[str, List[str]] = {}
    for d in dirs:
        subs = os.listdir(d)
        for sub in subs:
            subj = os.path.join(d, sub)
            if os.path.isdir(subj):
                voices[sub] = list(glob(f"{subj}/*.npz"))
                # fetch audio files if no npz files are found
                if len(voices[sub]) == 0:
                    voices[sub] = list(glob(f"{subj}/*.wav")) + list(glob(f"{subj}/*.mp3"))
    return voices


def load_npz(npz_file):
    x_history = np.load(npz_file)
    semantic = x_history["semantic_prompt"]
    coarse = x_history["coarse_prompt"]
    fine = x_history["fine_prompt"]
    return semantic, coarse, fine


def load_voice(model, voice: str, extra_voice_dirs: List[str] = []):  # pylint: disable=dangerous-default-value
    if voice == "random":
        return None, None, None

    voices = get_voices(extra_voice_dirs)
    paths = voices[voice]

    # bark only uses a single sample for cloning
    if len(paths) > 1:
        raise ValueError(f"Voice {voice} has multiple paths: {paths}")

    try:
        path = voices[voice]
    except KeyError as e:
        raise KeyError(f"Voice {voice} not found in {extra_voice_dirs}") from e

    if len(paths) == 1 and paths[0].endswith(".npz"):
        return load_npz(path[0])

    audio_path = paths[0]
    # replace the file extension with .npz
    output_path = os.path.splitext(audio_path)[0] + ".npz"
    generate_voice(audio=audio_path, model=model, output_path=output_path)
    return load_voice(model, voice, extra_voice_dirs)


def zero_crossing_rate(audio, frame_length=1024, hop_length=512):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) / 2)
    total_frames = 1 + int((len(audio) - frame_length) / hop_length)
    return zero_crossings / total_frames


def compute_spectral_contrast(audio_data, sample_rate, n_bands=6, fmin=200.0):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate, n_bands=n_bands, fmin=fmin)
    return np.mean(spectral_contrast)


def compute_average_bass_energy(audio_data, sample_rate, max_bass_freq=250):
    stft = librosa.stft(audio_data)
    power_spectrogram = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=stft.shape[0])
    bass_mask = frequencies <= max_bass_freq
    bass_energy = power_spectrogram[np.ix_(bass_mask, np.arange(power_spectrogram.shape[1]))].mean()
    return bass_energy


def generate_voice(
    audio,
    model,
    output_path,
):
    """Generate a new voice from a given audio and text prompt.

    Args:
        audio (np.ndarray): The audio to use as a base for the new voice.
        text (str): Transcription of the audio you are clonning.
        model (BarkModel): The BarkModel to use for generating the new voice.
        output_path (str): The path to save the generated voice to.
    """
    if isinstance(audio, str):
        audio, sr = torchaudio.load(audio)
        audio = convert_audio(audio, sr, model.config.sample_rate, model.encodec.channels)
        audio = audio.unsqueeze(0).to(model.device)

    with torch.no_grad():
        encoded_frames = model.encodec.encode(audio)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

    # move codes to cpu
    codes = codes.cpu().numpy()

    # generate semantic tokens
    # Load the HuBERT model
    hubert_manager = HubertManager()
    # hubert_manager.make_sure_hubert_installed(model_path=model.config.LOCAL_MODEL_PATHS["hubert"])
    hubert_manager.make_sure_tokenizer_installed(model_path=model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"])

    hubert_model = CustomHubert(checkpoint_path=model.config.LOCAL_MODEL_PATHS["hubert"]).to(model.device)

    # Load the CustomTokenizer model
    tokenizer = HubertTokenizer.load_from_checkpoint(
        model.config.LOCAL_MODEL_PATHS["hubert_tokenizer"], map_location=model.device
    )
    # semantic_tokens = model.text_to_semantic(
    #     text, max_gen_duration_s=seconds, top_k=50, top_p=0.95, temp=0.7
    # )  # not 100%
    semantic_vectors = hubert_model.forward(audio[0], input_sample_hz=model.config.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    semantic_tokens = semantic_tokens.cpu().numpy()

    np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)


def generate_text_semantic(
    text,
    model,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    base=None,
    use_kv_caching=True,
    **kwargs,  # pylint: disable=unused-argument
):
    """Generate semantic tokens from text.

    Args:
        text (str): The text to generate semantic tokens from.
        model (BarkModel): The BarkModel to use for generating the semantic tokens.
        history_prompt (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp (float): The temperature to use for the generation.
        top_k (int): The number of top tokens to consider for the generation.
        top_p (float): The cumulative probability to consider for the generation.
        silent (bool): Whether to silence the tqdm progress bar.
        min_eos_p (float): The minimum probability to consider for the end of sentence token.
        max_gen_duration_s (float): The maximum duration in seconds to generate for.
        allow_early_stop (bool): Whether to allow the generation to stop early.
        base (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.
        use_kv_caching (bool): Whether to use key-value caching for the generation.
        **kwargs: Additional keyword arguments. They are ignored.

    Returns:
        np.ndarray: The generated semantic tokens.
    """
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if all(v is not None for v in history_prompt) or base is not None:
        if history_prompt is not None:
            semantic_history = history_prompt[0]
        if base is not None:
            semantic_history = base[0]
        assert (
            isinstance(semantic_history, np.ndarray)
            and len(semantic_history.shape) == 1
            and len(semantic_history) > 0
            and semantic_history.min() >= 0
            and semantic_history.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    encoded_text = np.array(_tokenize(model.tokenizer, text)) + model.config.TEXT_ENCODING_OFFSET
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=model.config.TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=model.config.SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([model.config.SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([encoded_text, semantic_history, np.array([model.config.SEMANTIC_INFER_TOKEN])]).astype(np.int64)
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with inference_mode():
        x = x.to(model.device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=100)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model.semantic_model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, : model.config.SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [model.config.SEMANTIC_PAD_TOKEN]])
                )  # eos
            if top_p is not None:
                # faster to convert to numpy
                logits_device = relevant_logits.device
                logits_dtype = relevant_logits.type()
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = torch.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1)
            if allow_early_stop and (
                item_next == model.config.SEMANTIC_VOCAB_SIZE or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(100 - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / model.config.SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(100 - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(100 - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
            if req_pbar_state > pbar_state:
                pbar.update(req_pbar_state - pbar_state)
            pbar_state = req_pbar_state
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    assert all(out >= 0) and all(out < model.config.SEMANTIC_VOCAB_SIZE)
    clear_cuda_cache()
    return out


def _flatten_codebooks(arr, offset_size):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


def generate_coarse(
    x_semantic,
    model,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    max_coarse_history=630,  # min 60 (faster), max 630 (more context)
    sliding_window_len=60,
    base=None,
    use_kv_caching=True,
):
    """Generate coarse audio codes from semantic tokens.

    Args:
        x_semantic (np.ndarray): The semantic tokens to generate coarse audio codes from.
        model (BarkModel): The BarkModel to use for generating the coarse audio codes.
        history_prompt (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp (float): The temperature to use for the generation.
        top_k (int): The number of top tokens to consider for the generation.
        top_p (float): The cumulative probability to consider for the generation.
        silent (bool): Whether to silence the tqdm progress bar.
        max_coarse_history (int): The maximum number of coarse audio codes to use as history.
        sliding_window_len (int): The length of the sliding window to use for the generation.
        base (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.
        use_kv_caching (bool): Whether to use key-value caching for the generation.

    Returns:
        np.ndarray: The generated coarse audio codes.
    """
    assert (
        isinstance(x_semantic, np.ndarray)
        and len(x_semantic.shape) == 1
        and len(x_semantic) > 0
        and x_semantic.min() >= 0
        and x_semantic.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = (
        model.config.COARSE_RATE_HZ / model.config.SEMANTIC_RATE_HZ * model.config.N_COARSE_CODEBOOKS
    )
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if all(v is not None for v in history_prompt) or base is not None:
        if history_prompt is not None:
            x_history = history_prompt
            x_semantic_history = x_history[0]
            x_coarse_history = x_history[1]
        if base is not None:
            x_semantic_history = base[0]
            x_coarse_history = base[1]
        assert (
            isinstance(x_semantic_history, np.ndarray)
            and len(x_semantic_history.shape) == 1
            and len(x_semantic_history) > 0
            and x_semantic_history.min() >= 0
            and x_semantic_history.max() <= model.config.SEMANTIC_VOCAB_SIZE - 1
            and isinstance(x_coarse_history, np.ndarray)
            and len(x_coarse_history.shape) == 2
            and x_coarse_history.shape[0] == model.config.N_COARSE_CODEBOOKS
            and x_coarse_history.shape[-1] >= 0
            and x_coarse_history.min() >= 0
            and x_coarse_history.max() <= model.config.CODEBOOK_SIZE - 1
            and (
                round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                == round(semantic_to_coarse_ratio / model.config.N_COARSE_CODEBOOKS, 1)
            )
        )
        x_coarse_history = (
            _flatten_codebooks(x_coarse_history, model.config.CODEBOOK_SIZE) + model.config.SEMANTIC_VOCAB_SIZE
        )
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / model.config.N_COARSE_CODEBOOKS)
            * model.config.N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % model.config.N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(model.device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(model.device)
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                model.config.COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([model.config.COARSE_INFER_TOKEN])[None].to(model.device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % model.config.N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = model.coarse_model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = (
                    model.config.SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * model.config.CODEBOOK_SIZE
                )
                logit_end_idx = model.config.SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * model.config.CODEBOOK_SIZE
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    # faster to convert to numpy
                    logits_device = relevant_logits.device
                    logits_dtype = relevant_logits.type()
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(torch.nn.functional.softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = torch.nn.functional.softmax(relevant_logits / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history) :]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = (
        gen_coarse_arr.reshape(-1, model.config.N_COARSE_CODEBOOKS).T - model.config.SEMANTIC_VOCAB_SIZE
    )
    for n in range(1, model.config.N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * model.config.CODEBOOK_SIZE
    clear_cuda_cache()
    return gen_coarse_audio_arr


def generate_fine(
    x_coarse_gen,
    model,
    history_prompt=None,
    temp=0.5,
    silent=True,
    base=None,
):
    """Generate full audio codes from coarse audio codes.

    Args:
        x_coarse_gen (np.ndarray): The coarse audio codes to generate full audio codes from.
        model (BarkModel): The BarkModel to use for generating the full audio codes.
        history_prompt (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a prompt for the generation.
        temp (float): The temperature to use for the generation.
        silent (bool): Whether to silence the tqdm progress bar.
        base (tuple): A tuple of (semantic_history, coarse_history, fine_history) to use as a base for the generation.

    Returns:
        np.ndarray: The generated full audio codes.
    """
    assert (
        isinstance(x_coarse_gen, np.ndarray)
        and len(x_coarse_gen.shape) == 2
        and 1 <= x_coarse_gen.shape[0] <= model.config.N_FINE_CODEBOOKS - 1
        and x_coarse_gen.shape[1] > 0
        and x_coarse_gen.min() >= 0
        and x_coarse_gen.max() <= model.config.CODEBOOK_SIZE - 1
    )
    if all(v is not None for v in history_prompt) or base is not None:
        if history_prompt is not None:
            x_fine_history = history_prompt[2]
        if base is not None:
            x_fine_history = base[2]
        assert (
            isinstance(x_fine_history, np.ndarray)
            and len(x_fine_history.shape) == 2
            and x_fine_history.shape[0] == model.config.N_FINE_CODEBOOKS
            and x_fine_history.shape[1] >= 0
            and x_fine_history.min() >= 0
            and x_fine_history.max() <= model.config.CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((model.config.N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + model.config.CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((model.config.N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32)
                + model.config.CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with inference_mode():
        in_arr = torch.tensor(in_arr.T).to(model.device)
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, model.config.N_FINE_CODEBOOKS):
                logits = model.fine_model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, : model.config.CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, : model.config.CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    codebook_preds = torch.hstack(
                        [torch.multinomial(probs[n], num_samples=1) for n in range(rel_start_fill_idx, 1024)]
                    )
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, model.config.N_FINE_CODEBOOKS):
                in_arr[start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn] = in_buffer[
                    0, rel_start_fill_idx:, nn
                ]
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    clear_cuda_cache()
    return gen_fine_arr


def codec_decode(fine_tokens, model):
    """Turn quantized audio codes into audio array using encodec."""
    arr = torch.from_numpy(fine_tokens)[None]
    arr = arr.to(model.device)
    arr = arr.transpose(0, 1)
    emb = model.encodec.quantizer.decode(arr)
    out = model.encodec.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()
    return audio_arr
