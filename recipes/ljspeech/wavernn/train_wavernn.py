import os

from TTS.trainer import Trainer, TrainingArgs, init_training
from TTS.vocoder.configs import WavernnConfig

output_path = os.path.dirname(os.path.abspath(__file__))
config = WavernnConfig(
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000,
    seq_len=1280,
    pad_short=2000,
    use_noise_augment=False,
    eval_split_size=10,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    lr=1e-4,
    grad_clip=4,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)
args, config, output_path, _, c_logger, dashboard_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, dashboard_logger, cudnn_benchmark=True)
trainer.fit()
