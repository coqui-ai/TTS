from TTS.enhancer.config.base_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.bwe import BWE
from TTS.config import load_config
import torch
from librosa.core import load
from soundfile import write

path = "/home/julian/workspace/bwe"
config_path = path + "/config.json"
model_path = path + "/checkpoint_220000.pth"
wav_path = path + "/wavs/ESD_0012_Sad.wav"

config = BaseEnhancerConfig()
config.load_json(config_path)
model = BWE.init_from_config(config)
model.load_state_dict(torch.load(model_path)["model"])
model.eval()

wav = load(wav_path, sr=config.input_sr)
input = torch.from_numpy(wav[0])
input = input[:min(input.shape[0], 16000*6)].unsqueeze(0)
output = model.inference(input)["y_hat"]
output = output.squeeze(0).squeeze(0).detach().numpy()
write(path + "/output.wav", output, config.target_sr)
