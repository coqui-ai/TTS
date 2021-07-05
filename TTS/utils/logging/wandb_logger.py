from pathlib import Path

try:
    import wandb
    from wandb import init, finish
except ImportError:
    wandb = None


class WandbLogger:
    def __init__(self, **kwargs):
        self.run = None
        if wandb:
            self.run = wandb.init(**kwargs) if not wandb.run else wandb.run
        self.log_dict = {}

    def log(self, log_dict, prefix="", flush=False):
        """
        This function accumulates data in self.log_dict. If flush is set.
        the accumulated metrics will be logged directly to wandb dashboard.
        """
        for key, value in log_dict.items():
            self.log_dict[prefix + key] = value
        if flush:  # for cases where you don't want to accumulate data
            self.flush()

    def log_scalars(self, log_dict, prefix=""):
        if not self.run:
            return

        for key, value in log_dict.items():
            self.log_dict[prefix + key] = value

    def log_audios(self, log_dict, sample_rate, prefix=""):
        if not self.run:
            return

        prefix = "audios/" + prefix
        for key, value in log_dict.items():
            self.log_dict[prefix + key] = wandb.Audio(value, sample_rate=int(sample_rate))

    def log_figures(self, log_dict, prefix=""):
        if not self.run:
            return

        prefix = "figures/" + prefix
        for key, value in log_dict.items():
            self.log_dict[prefix + key] = wandb.Image(value)

    def flush(self):
        if self.run:
            wandb.log(self.log_dict)
        self.log_dict = {}

    def finish(self):
        """
        Finish this W&B run
        """
        self.run.finish()

    def log_artifact(self, file_or_dir, name, type, aliases=[]):
        artifact = wandb.Artifact(name, type=type)
        data_path = Path(file_or_dir)
        if data_path.is_dir():
            artifact.add_dir(str(data_path))
        elif data_path.is_file():
            artifact.add_file(str(data_path))

        self.run.log_artifact(artifact, aliases=aliases)
