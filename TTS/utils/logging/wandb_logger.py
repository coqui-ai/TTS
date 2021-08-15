# pylint: disable=W0613

import traceback
from pathlib import Path

try:
    import wandb
    from wandb import finish, init  # pylint: disable=W0611
except ImportError:
    wandb = None


class WandbLogger:
    def __init__(self, **kwargs):

        if not wandb:
            raise Exception("install wandb using `pip install wandb` to use WandbLogger")

        self.run = None
        self.run = wandb.init(**kwargs) if not wandb.run else wandb.run
        self.model_name = self.run.config.model
        self.log_dict = {}

    def model_weights(self, model):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.dict_to_scalar("weights", {"layer{}-{}/value".format(layer_num, name): param.max()})
            else:
                self.dict_to_scalar("weights", {"layer{}-{}/max".format(layer_num, name): param.max()})
                self.dict_to_scalar("weights", {"layer{}-{}/min".format(layer_num, name): param.min()})
                self.dict_to_scalar("weights", {"layer{}-{}/mean".format(layer_num, name): param.mean()})
                self.dict_to_scalar("weights", {"layer{}-{}/std".format(layer_num, name): param.std()})
                self.log_dict["weights/layer{}-{}/param".format(layer_num, name)] = wandb.Histogram(param)
                self.log_dict["weights/layer{}-{}/grad".format(layer_num, name)] = wandb.Histogram(param.grad)
            layer_num += 1

    def dict_to_scalar(self, scope_name, stats):
        for key, value in stats.items():
            self.log_dict["{}/{}".format(scope_name, key)] = value

    def dict_to_figure(self, scope_name, figures):
        for key, value in figures.items():
            self.log_dict["{}/{}".format(scope_name, key)] = wandb.Image(value)

    def dict_to_audios(self, scope_name, audios, sample_rate):
        for key, value in audios.items():
            if value.dtype == "float16":
                value = value.astype("float32")
            try:
                self.log_dict["{}/{}".format(scope_name, key)] = wandb.Audio(value, sample_rate=sample_rate)
            except RuntimeError:
                traceback.print_exc()

    def log(self, log_dict, prefix="", flush=False):
        for key, value in log_dict.items():
            self.log_dict[prefix + key] = value
        if flush:  # for cases where you don't want to accumulate data
            self.flush()

    def train_step_stats(self, step, stats):
        self.dict_to_scalar(f"{self.model_name}_TrainIterStats", stats)

    def train_epoch_stats(self, step, stats):
        self.dict_to_scalar(f"{self.model_name}_TrainEpochStats", stats)

    def train_figures(self, step, figures):
        self.dict_to_figure(f"{self.model_name}_TrainFigures", figures)

    def train_audios(self, step, audios, sample_rate):
        self.dict_to_audios(f"{self.model_name}_TrainAudios", audios, sample_rate)

    def eval_stats(self, step, stats):
        self.dict_to_scalar(f"{self.model_name}_EvalStats", stats)

    def eval_figures(self, step, figures):
        self.dict_to_figure(f"{self.model_name}_EvalFigures", figures)

    def eval_audios(self, step, audios, sample_rate):
        self.dict_to_audios(f"{self.model_name}_EvalAudios", audios, sample_rate)

    def test_audios(self, step, audios, sample_rate):
        self.dict_to_audios(f"{self.model_name}_TestAudios", audios, sample_rate)

    def test_figures(self, step, figures):
        self.dict_to_figure(f"{self.model_name}_TestFigures", figures)

    def add_text(self, title, text, step):
        pass

    def flush(self):
        if self.run:
            wandb.log(self.log_dict)
        self.log_dict = {}

    def finish(self):
        if self.run:
            self.run.finish()

    def log_artifact(self, file_or_dir, name, artifact_type, aliases=None):
        if not self.run:
            return
        name = "_".join([self.run.id, name])
        artifact = wandb.Artifact(name, type=artifact_type)
        data_path = Path(file_or_dir)
        if data_path.is_dir():
            artifact.add_dir(str(data_path))
        elif data_path.is_file():
            artifact.add_file(str(data_path))

        self.run.log_artifact(artifact, aliases=aliases)
