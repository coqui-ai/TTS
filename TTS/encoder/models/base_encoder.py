import numpy as np
import torch
import torchaudio
from coqpit import Coqpit
from torch import nn

from TTS.encoder.losses import AngleProtoLoss, GE2ELoss, SoftmaxAngleProtoLoss
from TTS.utils.generic_utils import set_init_dict
from TTS.utils.io import load_fsspec


class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2

        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class BaseEncoder(nn.Module):
    """Base `encoder` class. Every new `encoder` model must inherit this.

    It defines common `encoder` specific functions.
    """

    # pylint: disable=W0102
    def __init__(self):
        super(BaseEncoder, self).__init__()

    def get_torch_mel_spectrogram_class(self, audio_config):
        return torch.nn.Sequential(
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

    @torch.no_grad()
    def inference(self, x, l2_norm=True):
        return self.forward(x, l2_norm)

    @torch.no_grad()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        # map to the waveform size
        if self.use_torch_spec:
            num_frames = num_frames * self.audio_config["hop_length"]

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
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings

    def get_criterion(self, c: Coqpit, num_classes=None):
        if c.loss == "ge2e":
            criterion = GE2ELoss(loss_method="softmax")
        elif c.loss == "angleproto":
            criterion = AngleProtoLoss()
        elif c.loss == "softmaxproto":
            criterion = SoftmaxAngleProtoLoss(c.model_params["proj_dim"], num_classes)
        else:
            raise Exception("The %s  not is a loss supported" % c.loss)
        return criterion

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str,
        eval: bool = False,
        use_cuda: bool = False,
        criterion=None,
        cache=False,
    ):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        try:
            self.load_state_dict(state["model"])
            print(" > Model fully restored. ")
        except (KeyError, RuntimeError) as error:
            # If eval raise the error
            if eval:
                raise error

            print(" > Partial model initialization.")
            model_dict = self.state_dict()
            model_dict = set_init_dict(model_dict, state["model"], c)
            self.load_state_dict(model_dict)
            del model_dict

        # load the criterion for restore_path
        if criterion is not None and "criterion" in state:
            try:
                criterion.load_state_dict(state["criterion"])
            except (KeyError, RuntimeError) as error:
                print(" > Criterion load ignored because of:", error)

        # instance and load the criterion for the encoder classifier in inference time
        if (
            eval
            and criterion is None
            and "criterion" in state
            and getattr(config, "map_classid_to_classname", None) is not None
        ):
            criterion = self.get_criterion(config, len(config.map_classid_to_classname))
            criterion.load_state_dict(state["criterion"])

        if use_cuda:
            self.cuda()
            if criterion is not None:
                criterion = criterion.cuda()

        if eval:
            self.eval()
            assert not self.training

        if not eval:
            return criterion, state["step"]
        return criterion
