from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class SpeakerEncoderConfig(BaseTrainingConfig):
    """Defines parameters for Speaker Encoder model."""

    model: str = "speaker_encoder"
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # model params
    model_params: Dict = field(
        default_factory=lambda: {
            "model_name": "lstm",
            "input_dim": 80,
            "proj_dim": 256,
            "lstm_dim": 768,
            "num_lstm_layers": 3,
            "use_lstm_with_projection": True,
        }
    )

    audio_augmentation: Dict = field(default_factory=lambda: {})

    storage: Dict = field(
        default_factory=lambda: {
            "sample_from_storage_p": 0.66,  # the probability with which we'll sample from the DataSet in-memory storage
            "storage_size": 15,  # the size of the in-memory storage with respect to a single batch
        }
    )

    # training params
    max_train_step: int = 1000000  # end training when number of training steps reaches this value.
    loss: str = "angleproto"
    grad_clip: float = 3.0
    lr: float = 0.0001
    lr_decay: bool = False
    warmup_steps: int = 4000
    wd: float = 1e-6

    # logging params
    tb_model_param_stats: bool = False
    steps_plot_stats: int = 10
    checkpoint: bool = True
    save_step: int = 1000
    print_step: int = 20

    # data loader
    num_speakers_in_batch: int = MISSING
    num_utters_per_speaker: int = MISSING
    num_loader_workers: int = MISSING
    skip_speakers: bool = False
    voice_len: float = 1.6

    def check_values(self):
        super().check_values()
        c = asdict(self)
        assert (
            c["model_params"]["input_dim"] == self.audio.num_mels
        ), " [!] model input dimendion must be equal to melspectrogram dimension."
