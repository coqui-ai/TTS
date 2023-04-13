import os
import urllib.request

import torch

from TTS.utils.generic_utils import get_user_data_dir
from TTS.vc.modules.freevc.wavlm.wavlm import WavLM, WavLMConfig

model_uri = "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/WavLM-Large.pt"


def get_wavlm(device="cpu"):
    """Download the model and return the model object."""

    output_path = get_user_data_dir("tts")

    output_path = os.path.join(output_path, "wavlm")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "WavLM-Large.pt")
    if not os.path.exists(output_path):
        print(f" > Downloading WavLM model to {output_path} ...")
        urllib.request.urlretrieve(model_uri, output_path)

    checkpoint = torch.load(output_path, map_location=torch.device(device))
    cfg = WavLMConfig(checkpoint["cfg"])
    wavlm = WavLM(cfg).to(device)
    wavlm.load_state_dict(checkpoint["model"])
    wavlm.eval()
    return wavlm


if __name__ == "__main__":
    wavlm = get_wavlm()
