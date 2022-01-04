import numpy as np
import torch
import torchaudio
from torch import nn

from TTS.speaker_encoder.models.resnet import PreEmphasis
from TTS.utils.io import load_fsspec


class LSTMWithProjection(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, proj_size, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)


class LSTMWithoutProjection(nn.Module):
    def __init__(self, input_dim, lstm_dim, proj_dim, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim, num_layers=num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(lstm_dim, proj_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.relu(self.linear(hidden[-1]))


class LSTMSpeakerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        proj_dim=256,
        lstm_dim=768,
        num_lstm_layers=3,
        use_lstm_with_projection=True,
        use_torch_spec=False,
        audio_config=None,
    ):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = torch.nn.Sequential(
                PreEmphasis(audio_config["preemphasis"]),
                # TorchSTFT(
                #     n_fft=audio_config["fft_size"],
                #     hop_length=audio_config["hop_length"],
                #     win_length=audio_config["win_length"],
                #     sample_rate=audio_config["sample_rate"],
                #     window="hamming_window",
                #     mel_fmin=0.0,
                #     mel_fmax=None,
                #     use_htk=True,
                #     do_amp_to_db=False,
                #     n_mels=audio_config["num_mels"],
                #     power=2.0,
                #     use_mel=True,
                #     mel_norm=None,
                # )
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=audio_config["sample_rate"],
                    n_fft=audio_config["fft_size"],
                    win_length=audio_config["win_length"],
                    hop_length=audio_config["hop_length"],
                    window_fn=torch.hamming_window,
                    n_mels=audio_config["num_mels"],
                ),
            )
        else:
            self.torch_spec = None

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, l2_norm=True):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_torch_spec:
                    x.squeeze_(1)
                    x = self.torch_spec(x)
                x = self.instancenorm(x).transpose(1, 2)
        d = self.layers(x)
        if self.use_lstm_with_projection:
            d = d[:, -1]
        if l2_norm:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        return d

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        d = self.forward(x, l2_norm=l2_norm)
        return d

    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)

        return embeddings

    def batch_compute_embedding(self, x, seq_lens, num_frames=160, overlap=0.5):
        """
        Generate embeddings for a batch of utterances
        x: BxTxD
        """
        num_overlap = num_frames * overlap
        max_len = x.shape[1]
        embed = None
        num_iters = seq_lens / (num_frames - num_overlap)
        cur_iter = 0
        for offset in range(0, max_len, num_frames - num_overlap):
            cur_iter += 1
            end_offset = min(x.shape[1], offset + num_frames)
            frames = x[:, offset:end_offset]
            if embed is None:
                embed = self.inference(frames)
            else:
                embed[cur_iter <= num_iters, :] += self.inference(frames[cur_iter <= num_iters, :, :])
        return embed / num_iters

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(self, config: dict, checkpoint_path: str, eval: bool = False, use_cuda: bool = False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if use_cuda:
            self.cuda()
        if eval:
            self.eval()
            assert not self.training
