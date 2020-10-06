import datetime
import os
import re

import torch
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.utils.generic_utils import check_argument


def to_camel(text):
    text = text.capitalize()
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)


def setup_model(c):
    model = SpeakerEncoder(c.model['input_dim'], c.model['proj_dim'],
                           c.model['lstm_dim'], c.model['num_lstm_layers'])
    return model


def save_checkpoint(model, optimizer, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'loss': model_loss,
        'date': datetime.date.today().strftime("%B %d, %Y"),
    }
    torch.save(state, checkpoint_path)


def save_best_model(model, optimizer, model_loss, best_loss, out_path,
                    current_step):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            'model': new_state_dict,
            'optimizer': optimizer.state_dict(),
            'step': current_step,
            'loss': model_loss,
            'date': datetime.date.today().strftime("%B %d, %Y"),
        }
        best_loss = model_loss
        bestmodel_path = 'best_model.pth.tar'
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(
            model_loss, bestmodel_path))
        torch.save(state, bestmodel_path)
    return best_loss


def check_config_speaker_encoder(c):
    """Check the config.json file of the speaker encoder"""
    check_argument('run_name', c, restricted=True, val_type=str)
    check_argument('run_description', c, val_type=str)

    # audio processing parameters
    check_argument('audio', c, restricted=True, val_type=dict)
    check_argument('num_mels', c['audio'], restricted=True, val_type=int, min_val=10, max_val=2056)
    check_argument('fft_size', c['audio'], restricted=True, val_type=int, min_val=128, max_val=4058)
    check_argument('sample_rate', c['audio'], restricted=True, val_type=int, min_val=512, max_val=100000)
    check_argument('frame_length_ms', c['audio'], restricted=True, val_type=float, min_val=10, max_val=1000, alternative='win_length')
    check_argument('frame_shift_ms', c['audio'], restricted=True, val_type=float, min_val=1, max_val=1000, alternative='hop_length')
    check_argument('preemphasis', c['audio'], restricted=True, val_type=float, min_val=0, max_val=1)
    check_argument('min_level_db', c['audio'], restricted=True, val_type=int, min_val=-1000, max_val=10)
    check_argument('ref_level_db', c['audio'], restricted=True, val_type=int, min_val=0, max_val=1000)
    check_argument('power', c['audio'], restricted=True, val_type=float, min_val=1, max_val=5)
    check_argument('griffin_lim_iters', c['audio'], restricted=True, val_type=int, min_val=10, max_val=1000)

    # training parameters
    check_argument('loss', c, enum_list=['ge2e', 'angleproto'], restricted=True, val_type=str)
    check_argument('grad_clip', c, restricted=True, val_type=float)
    check_argument('epochs', c, restricted=True, val_type=int, min_val=1)
    check_argument('lr', c, restricted=True, val_type=float, min_val=0)
    check_argument('lr_decay', c, restricted=True, val_type=bool)
    check_argument('warmup_steps', c, restricted=True, val_type=int, min_val=0)
    check_argument('tb_model_param_stats', c, restricted=True, val_type=bool)
    check_argument('num_speakers_in_batch', c, restricted=True, val_type=int)
    check_argument('num_loader_workers', c, restricted=True, val_type=int)
    check_argument('wd', c, restricted=True, val_type=float, min_val=0.0, max_val=1.0)

    # checkpoint and output parameters
    check_argument('steps_plot_stats', c, restricted=True, val_type=int)
    check_argument('checkpoint', c, restricted=True, val_type=bool)
    check_argument('save_step', c, restricted=True, val_type=int)
    check_argument('print_step', c, restricted=True, val_type=int)
    check_argument('output_path', c, restricted=True, val_type=str)

    # model parameters
    check_argument('model', c, restricted=True, val_type=dict)
    check_argument('input_dim', c['model'], restricted=True, val_type=int)
    check_argument('proj_dim', c['model'], restricted=True, val_type=int)
    check_argument('lstm_dim', c['model'], restricted=True, val_type=int)
    check_argument('num_lstm_layers', c['model'], restricted=True, val_type=int)
    check_argument('use_lstm_with_projection', c['model'], restricted=True, val_type=bool)

    # in-memory storage parameters
    check_argument('storage', c, restricted=True, val_type=dict)
    check_argument('sample_from_storage_p', c['storage'], restricted=True, val_type=float, min_val=0.0, max_val=1.0)
    check_argument('storage_size', c['storage'], restricted=True, val_type=int, min_val=1, max_val=100)
    check_argument('additive_noise', c['storage'], restricted=True, val_type=float, min_val=0.0, max_val=1.0)

    # datasets - checking only the first entry
    check_argument('datasets', c, restricted=True, val_type=list)
    for dataset_entry in c['datasets']:
        check_argument('name', dataset_entry, restricted=True, val_type=str)
        check_argument('path', dataset_entry, restricted=True, val_type=str)
        check_argument('meta_file_train', dataset_entry, restricted=True, val_type=[str, list])
        check_argument('meta_file_val', dataset_entry, restricted=True, val_type=str)

