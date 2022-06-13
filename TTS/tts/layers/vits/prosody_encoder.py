from TTS.tts.layers.tacotron.capacitron_layers import CapacitronVAE
from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.encoder.models.resnet import ResNetSpeakerEncoder

class VitsGST(GST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, input_lengths=None, speaker_embedding=None):
        style_embed = super().forward(inputs, speaker_embedding=speaker_embedding)
        return style_embed, None


class VitsVAE(CapacitronVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = None

    def forward(self, inputs, input_lengths=None):
        VAE_embedding, posterior_distribution, prior_distribution, _ = super().forward([inputs, input_lengths])
        return VAE_embedding.to(inputs.device), [posterior_distribution, prior_distribution]


class ResNetProsodyEncoder(ResNetSpeakerEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, input_lengths=None):
        style_embed = super().forward(inputs, l2_norm=True).unsqueeze(1)
        return style_embed, None