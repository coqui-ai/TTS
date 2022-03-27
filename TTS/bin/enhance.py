from TTS.enhancer.config.base_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.bwe import BWE
from TTS.config import load_config
import torch
from librosa.core import load
from soundfile import write

path = "/home/julian/workspace/enhancer_bwe_vctk-March-27-2022_03+27PM-a3a30a27/"
config_path = path + "/config.json"
model_path = path + "/checkpoint_1250.pth"
wav_path = "/home/julian/workspace/dujardin.wav"

config = BaseEnhancerConfig()
config.load_json(config_path)
model = BWE.init_from_config(config)
model.load_state_dict(torch.load(model_path)["model"])
model.eval()

wav = load(wav_path, sr=config.input_sr)
input = torch.from_numpy(wav[0]).unsqueeze(0)
output = model.inference(input)["y_hat"]
output = output.squeeze(0).squeeze(0).detach().numpy()
write(path + "/output.wav", output, config.target_sr)
