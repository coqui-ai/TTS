import os
import glob
import torch
import shutil
import datetime
import subprocess
import importlib
import numpy as np
from collections import Counter


def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n")
                       if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    return current


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # Not copying .git folder into docker container
    except subprocess.CalledProcessError:
        commit = "0000000"
    print(' > Git Hash: {}'.format(commit))
    return commit


def create_experiment_folder(root_path, model_name, debug):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    if debug:
        commit_hash = 'debug'
    else:
        commit_hash = get_commit_hash()
    output_folder = os.path.join(
        root_path, model_name + '-' + date_str + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    print(" > Experiment folder: {}".format(output_folder))
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = glob.glob(experiment_path + "/*.pth.tar")
    if not checkpoint_files:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path, ignore_errors=True)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(items):
    is_multi_speaker = False
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = 500 if len(items) * 0.01 > 500 else int(
        len(items) * 0.01)
    assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        # most stupid code ever -- Fix it !
        while len(items_eval) < eval_split_size:
            speakers = [item[-1] for item in items]
            speaker_counter = Counter(speakers)
            item_idx = np.random.randint(0, len(items))
            if speaker_counter[items[item_idx][-1]] > 1:
                items_eval.append(items[item_idx])
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint_state.items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict),
                                                     len(model_dict)))
    return model_dict


def setup_model(num_chars, num_speakers, c):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('TTS.models.' + c.model.lower())
    MyModel = getattr(MyModel, c.model)
    if c.model.lower() in "tacotron":
        model = MyModel(num_chars=num_chars,
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=int(c.audio['fft_size'] / 2 + 1),
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        gst_embedding_dim=c.gst['gst_embedding_dim'],
                        gst_num_heads=c.gst['gst_num_heads'],
                        gst_style_tokens=c.gst['gst_style_tokens'],
                        memory_size=c.memory_size,
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r)
    elif c.model.lower() == "tacotron2":
        model = MyModel(num_chars=num_chars,
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=c.audio['num_mels'],
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        gst_embedding_dim=c.gst['gst_embedding_dim'],
                        gst_num_heads=c.gst['gst_num_heads'],
                        gst_style_tokens=c.gst['gst_style_tokens'],
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r)
    return model

class KeepAverage():
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        else:
            # else update existing value
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * \
                    self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)


def _check_argument(name, c, enum_list=None, max_val=None, min_val=None, restricted=False, val_type=None, alternative=None):
    if alternative in c.keys() and c[alternative] is not None:
        return
    if restricted:
        assert name in c.keys(), f' [!] {name} not defined in config.json'
    if name in c.keys():
        if max_val:
            assert c[name] <= max_val, f' [!] {name} is larger than max value {max_val}'
        if min_val:
            assert c[name] >= min_val, f' [!] {name} is smaller than min value {min_val}'
        if enum_list:
            assert c[name].lower() in enum_list, f' [!] {name} is not a valid value'
        if val_type:
            assert isinstance(c[name], val_type) or c[name] is None, f' [!] {name} has wrong type - {type(c[name])} vs {val_type}'


def check_config(c):
    _check_argument('model', c, enum_list=['tacotron', 'tacotron2'], restricted=True, val_type=str)
    _check_argument('run_name', c, restricted=True, val_type=str)
    _check_argument('run_description', c, val_type=str)

    # AUDIO
    _check_argument('audio', c, restricted=True, val_type=dict)

    # audio processing parameters
    _check_argument('num_mels', c['audio'], restricted=True, val_type=int, min_val=10, max_val=2056)
    _check_argument('fft_size', c['audio'], restricted=True, val_type=int, min_val=128, max_val=4058)
    _check_argument('sample_rate', c['audio'], restricted=True, val_type=int, min_val=512, max_val=100000)
    _check_argument('frame_length_ms', c['audio'], restricted=True, val_type=float, min_val=10, max_val=1000, alternative='win_length')
    _check_argument('frame_shift_ms', c['audio'], restricted=True, val_type=float, min_val=1, max_val=1000, alternative='hop_length')
    _check_argument('preemphasis', c['audio'], restricted=True, val_type=float, min_val=0, max_val=1)
    _check_argument('min_level_db', c['audio'], restricted=True, val_type=int, min_val=-1000, max_val=10)
    _check_argument('ref_level_db', c['audio'], restricted=True, val_type=int, min_val=0, max_val=1000)
    _check_argument('power', c['audio'], restricted=True, val_type=float, min_val=1, max_val=5)
    _check_argument('griffin_lim_iters', c['audio'], restricted=True, val_type=int, min_val=10, max_val=1000)

    # vocabulary parameters
    _check_argument('characters', c, restricted=False, val_type=dict)
    _check_argument('pad', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    _check_argument('eos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    _check_argument('bos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    _check_argument('characters', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    _check_argument('phonemes', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    _check_argument('punctuations', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)

    # normalization parameters
    _check_argument('signal_norm', c['audio'], restricted=True, val_type=bool)
    _check_argument('symmetric_norm', c['audio'], restricted=True, val_type=bool)
    _check_argument('max_norm', c['audio'], restricted=True, val_type=float, min_val=0.1, max_val=1000)
    _check_argument('clip_norm', c['audio'], restricted=True, val_type=bool)
    _check_argument('mel_fmin', c['audio'], restricted=True, val_type=float, min_val=0.0, max_val=1000)
    _check_argument('mel_fmax', c['audio'], restricted=True, val_type=float, min_val=500.0)
    _check_argument('spec_gain', c['audio'], restricted=True, val_type=float, min_val=1, max_val=100)
    _check_argument('do_trim_silence', c['audio'], restricted=True, val_type=bool)
    _check_argument('trim_db', c['audio'], restricted=True, val_type=int)

    # training parameters
    _check_argument('batch_size', c, restricted=True, val_type=int, min_val=1)
    _check_argument('eval_batch_size', c, restricted=True, val_type=int, min_val=1)
    _check_argument('r', c, restricted=True, val_type=int, min_val=1)
    _check_argument('gradual_training', c, restricted=False, val_type=list)
    _check_argument('loss_masking', c, restricted=True, val_type=bool)
    # _check_argument('grad_accum', c, restricted=True, val_type=int, min_val=1, max_val=100)

    # validation parameters
    _check_argument('run_eval', c, restricted=True, val_type=bool)
    _check_argument('test_delay_epochs', c, restricted=True, val_type=int, min_val=0)
    _check_argument('test_sentences_file', c, restricted=False, val_type=str)

    # optimizer
    _check_argument('noam_schedule', c, restricted=False, val_type=bool)
    _check_argument('grad_clip', c, restricted=True, val_type=float, min_val=0.0)
    _check_argument('epochs', c, restricted=True, val_type=int, min_val=1)
    _check_argument('lr', c, restricted=True, val_type=float, min_val=0)
    _check_argument('wd', c, restricted=True, val_type=float, min_val=0)
    _check_argument('warmup_steps', c, restricted=True, val_type=int, min_val=0)
    _check_argument('seq_len_norm', c, restricted=True, val_type=bool)

    # tacotron prenet
    _check_argument('memory_size', c, restricted=True, val_type=int, min_val=-1)
    _check_argument('prenet_type', c, restricted=True, val_type=str, enum_list=['original', 'bn'])
    _check_argument('prenet_dropout', c, restricted=True, val_type=bool)

    # attention
    _check_argument('attention_type', c, restricted=True, val_type=str, enum_list=['graves', 'original'])
    _check_argument('attention_heads', c, restricted=True, val_type=int)
    _check_argument('attention_norm', c, restricted=True, val_type=str, enum_list=['sigmoid', 'softmax'])
    _check_argument('windowing', c, restricted=True, val_type=bool)
    _check_argument('use_forward_attn', c, restricted=True, val_type=bool)
    _check_argument('forward_attn_mask', c, restricted=True, val_type=bool)
    _check_argument('transition_agent', c, restricted=True, val_type=bool)
    _check_argument('transition_agent', c, restricted=True, val_type=bool)
    _check_argument('location_attn', c, restricted=True, val_type=bool)
    _check_argument('bidirectional_decoder', c, restricted=True, val_type=bool)
    _check_argument('double_decoder_consistency', c, restricted=True, val_type=bool)
    _check_argument('ddc_r', c, restricted='double_decoder_consistency' in c.keys(), min_val=1, max_val=7, val_type=int)

    # stopnet
    _check_argument('stopnet', c, restricted=True, val_type=bool)
    _check_argument('separate_stopnet', c, restricted=True, val_type=bool)

    # tensorboard
    _check_argument('print_step', c, restricted=True, val_type=int, min_val=1)
    _check_argument('tb_plot_step', c, restricted=True, val_type=int, min_val=1)
    _check_argument('save_step', c, restricted=True, val_type=int, min_val=1)
    _check_argument('checkpoint', c, restricted=True, val_type=bool)
    _check_argument('tb_model_param_stats', c, restricted=True, val_type=bool)

    # dataloading
    # pylint: disable=import-outside-toplevel
    from TTS.utils.text import cleaners
    _check_argument('text_cleaner', c, restricted=True, val_type=str, enum_list=dir(cleaners))
    _check_argument('enable_eos_bos_chars', c, restricted=True, val_type=bool)
    _check_argument('num_loader_workers', c, restricted=True, val_type=int, min_val=0)
    _check_argument('num_val_loader_workers', c, restricted=True, val_type=int, min_val=0)
    _check_argument('batch_group_size', c, restricted=True, val_type=int, min_val=0)
    _check_argument('min_seq_len', c, restricted=True, val_type=int, min_val=0)
    _check_argument('max_seq_len', c, restricted=True, val_type=int, min_val=10)

    # paths
    _check_argument('output_path', c, restricted=True, val_type=str)

    # multi-speaker
    _check_argument('use_speaker_embedding', c, restricted=True, val_type=bool)

    # GST
    _check_argument('use_gst', c, restricted=True, val_type=bool)
    _check_argument('gst', c, restricted=True, val_type=dict)
    _check_argument('gst_style_input', c['gst'], restricted=True, val_type=str)
    _check_argument('gst_embedding_dim', c['gst'], restricted=True, val_type=int, min_val=1)
    _check_argument('gst_num_heads', c['gst'], restricted=True, val_type=int, min_val=1)
    _check_argument('gst_style_tokens', c['gst'], restricted=True, val_type=int, min_val=1)

    # datasets - checking only the first entry
    _check_argument('datasets', c, restricted=True, val_type=list)
    for dataset_entry in c['datasets']:
        _check_argument('name', dataset_entry, restricted=True, val_type=str)
        _check_argument('path', dataset_entry, restricted=True, val_type=str)
        _check_argument('meta_file_train', dataset_entry, restricted=True, val_type=str)
        _check_argument('meta_file_val', dataset_entry, restricted=True, val_type=str)
