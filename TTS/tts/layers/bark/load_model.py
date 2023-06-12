import contextlib

# import funcy
import functools
import hashlib
import logging
import os
import re

import requests
import torch
import tqdm
from encodec import EncodecModel
from transformers import BertTokenizer

from TTS.tts.layers.bark.model import GPT, GPTConfig
from TTS.tts.layers.bark.model_fine import FineGPT, FineGPTConfig

if (
    torch.cuda.is_available()
    and hasattr(torch.cuda, "amp")
    and hasattr(torch.cuda.amp, "autocast")
    and torch.cuda.is_bf16_supported()
):
    autocast = functools.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:

    @contextlib.contextmanager
    def autocast():
        yield


# hold models in global scope to lazy load
global models
models = {}

logger = logging.getLogger(__name__)


if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    logger.warning(
        "torch version does not support flash attention. You will get significantly faster"
        + " inference speed by upgrade torch to newest version / nightly."
    )


def _string_md5(s):
    m = hashlib.md5()
    m.update(s.encode("utf-8"))
    return m.hexdigest()


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _get_ckpt_path(model_type, CACHE_DIR):
    model_name = _string_md5(REMOTE_MODEL_PATHS[model_type]["path"])
    return os.path.join(CACHE_DIR, f"{model_name}.pt")


S3_BUCKET_PATH_RE = r"s3\:\/\/(.+?)\/"


def _parse_s3_filepath(s3_filepath):
    bucket_name = re.search(S3_BUCKET_PATH_RE, s3_filepath).group(1)
    rel_s3_filepath = re.sub(S3_BUCKET_PATH_RE, "", s3_filepath)
    return bucket_name, rel_s3_filepath


def _download(from_s3_path, to_local_path, CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
    response = requests.get(from_s3_path, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(to_local_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise ValueError("ERROR, something went wrong")


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_models(model_key=None):
    global models
    model_keys = [model_key] if model_key is not None else models.keys()
    for k in model_keys:
        if k in models:
            del models[k]
    _clear_cuda_cache()


def _load_model(ckpt_path, device, config, model_type="text"):
    logger.info(f"loading {model_type} model from {ckpt_path}...")

    if device == "cpu":
        logger.warning("No GPU being used. Careful, Inference might be extremely slow!")
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    if (
        not config.USE_SMALLER_MODELS
        and os.path.exists(ckpt_path)
        and _md5(ckpt_path) != config.REMOTE_MODEL_PATHS[model_type]["checksum"]
    ):
        logger.warning(f"found outdated {model_type} model, removing...")
        os.remove(ckpt_path)
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading...")
        _download(config.REMOTE_MODEL_PATHS[model_type]["path"], ckpt_path, config.CACHE_DIR)

    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]

    gptconf = ConfigClass(**checkpoint["model_args"])
    if model_type == "text":
        config.semantic_config = gptconf
    elif model_type == "coarse":
        config.coarse_config = gptconf
    elif model_type == "fine":
        config.fine_config = gptconf

    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict
    _clear_cuda_cache()
    return model, config


def _load_codec_model(device):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    model.to(device)
    _clear_cuda_cache()
    return model


def load_model(ckpt_path=None, use_gpu=True, force_reload=False, model_type="text"):
    _load_model_f = functools.partial(_load_model, model_type=model_type)
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    if torch.cuda.device_count() == 0 or not use_gpu:
        device = "cpu"
    else:
        device = "cuda"
    model_key = str(device) + f"__{model_type}"
    if model_key not in models or force_reload:
        if ckpt_path is None:
            ckpt_path = _get_ckpt_path(model_type)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    return models[model_key]


def load_codec_model(use_gpu=True, force_reload=False):
    global models
    if torch.cuda.device_count() == 0 or not use_gpu:
        device = "cpu"
    else:
        device = "cuda"
    model_key = str(device) + f"__codec"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = _load_codec_model(device)
        models[model_key] = model
    return models[model_key]


def preload_models(
    text_ckpt_path=None, coarse_ckpt_path=None, fine_ckpt_path=None, use_gpu=True, use_smaller_models=False
):
    global USE_SMALLER_MODELS
    global REMOTE_MODEL_PATHS
    if use_smaller_models:
        USE_SMALLER_MODELS = True
        logger.info("Using smaller models generation.py")
        REMOTE_MODEL_PATHS = SMALL_REMOTE_MODEL_PATHS

    _ = load_model(ckpt_path=text_ckpt_path, model_type="text", use_gpu=use_gpu, force_reload=True)
    _ = load_model(ckpt_path=coarse_ckpt_path, model_type="coarse", use_gpu=use_gpu, force_reload=True)
    _ = load_model(ckpt_path=fine_ckpt_path, model_type="fine", use_gpu=use_gpu, force_reload=True)
    _ = load_codec_model(use_gpu=use_gpu, force_reload=True)
