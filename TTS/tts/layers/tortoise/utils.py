import os
try: import gdown
except ImportError:
    raise ImportError(
        "Sorry, gdown is required in order to download the new BigVGAN vocoder.\n"
        "Please install it with `pip install gdown` and try again."
    )
from urllib import request

import progressbar

D_STEM = "https://drive.google.com/uc?id="

DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "tortoise", "models"
)
MODELS_DIR = os.environ.get("TORTOISE_MODELS_DIR", DEFAULT_MODELS_DIR)
MODELS = {
    "autoregressive.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth",
    "classifier.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth",
    "clvp2.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth",
    "cvvp.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth",
    "diffusion_decoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth",
    "vocoder.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth",
    "rlg_auto.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth",
    "rlg_diffuser.pth": "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth",
    # these links are from the nvidia gdrive
    "bigvgan_base_24khz_100band_g.pth": "https://drive.google.com/uc?id=1_cKskUDuvxQJUEBwdgjAxKuDTUW6kPdY",
    "bigvgan_24khz_100band_g.pth": "https://drive.google.com/uc?id=1wmP_mAs7d00KHVfVEl8B5Gb72Kzpcavp",
}

pbar = None
def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None

    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            continue
        print(f"Downloading {model_name} from {url}...")
        if D_STEM in url:
            gdown.download(url, model_path, quiet=False)
        else:
            request.urlretrieve(url, model_path, show_progress)
        print("Done.")

def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path) and models_dir == MODELS_DIR:
        download_models([model_name])
    return model_path