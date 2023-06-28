import contextlib
import functools
import hashlib
import logging
import os

import requests
import torch
import tqdm

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

logger = logging.getLogger(__name__)


if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    logger.warning(
        "torch version does not support flash attention. You will get significantly faster"
        + " inference speed by upgrade torch to newest version / nightly."
    )


def _md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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
    if total_size_in_bytes not in [0, progress_bar.n]:
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
def inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(ckpt_path, device, config, model_type="text"):
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
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set(k for k in extra_keys if not k.endswith(".attn.bias"))
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set(k for k in missing_keys if not k.endswith(".attn.bias"))
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
    clear_cuda_cache()
    return model, config
