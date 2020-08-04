import traceback
from tensorboardX import SummaryWriter


class TensorboardLogger(object):
    def __init__(self, log_dir, model_name):
        self.model_name = model_name
        self.writer = SummaryWriter(log_dir)
        self.train_stats = {}
        self.eval_stats = {}

    def tb_model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            if param.numel() == 1:
                self.writer.add_scalar(
                    "layer{}-{}/value".format(layer_num, name),
                    param.max(), step)
            else:
                self.writer.add_scalar(
                    "layer{}-{}/max".format(layer_num, name),
                    param.max(), step)
                self.writer.add_scalar(
                    "layer{}-{}/min".format(layer_num, name),
                    param.min(), step)
                self.writer.add_scalar(
                    "layer{}-{}/mean".format(layer_num, name),
                    param.mean(), step)
                self.writer.add_scalar(
                    "layer{}-{}/std".format(layer_num, name),
                    param.std(), step)
                self.writer.add_histogram(
                    "layer{}-{}/param".format(layer_num, name), param, step)
                self.writer.add_histogram(
                    "layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def dict_to_tb_scalar(self, scope_name, stats, step):
        for key, value in stats.items():
            self.writer.add_scalar('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_figure(self, scope_name, figures, step):
        for key, value in figures.items():
            self.writer.add_figure('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            try:
                self.writer.add_audio('{}/{}'.format(scope_name, key), value, step, sample_rate=sample_rate)
            except RuntimeError:
                traceback.print_exc()

    def tb_train_iter_stats(self, step, stats):
        self.dict_to_tb_scalar(f"{self.model_name}_TrainIterStats", stats, step)

    def tb_train_epoch_stats(self, step, stats):
        self.dict_to_tb_scalar(f"{self.model_name}_TrainEpochStats", stats, step)

    def tb_train_figures(self, step, figures):
        self.dict_to_tb_figure(f"{self.model_name}_TrainFigures", figures, step)

    def tb_train_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios(f"{self.model_name}_TrainAudios", audios, step, sample_rate)

    def tb_eval_stats(self, step, stats):
        self.dict_to_tb_scalar(f"{self.model_name}_EvalStats", stats, step)

    def tb_eval_figures(self, step, figures):
        self.dict_to_tb_figure(f"{self.model_name}_EvalFigures", figures, step)

    def tb_eval_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios(f"{self.model_name}_EvalAudios", audios, step, sample_rate)

    def tb_test_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios(f"{self.model_name}_TestAudios", audios, step, sample_rate)

    def tb_test_figures(self, step, figures):
        self.dict_to_tb_figure(f"{self.model_name}_TestFigures", figures, step)

    def tb_add_text(self, title, text, step):
        self.writer.add_text(title, text, step)
