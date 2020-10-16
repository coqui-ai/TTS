import os
import torch
import datetime
import pickle as pickle_tts

from TTS.utils.io import RenamingUnpickler



def load_checkpoint(model, checkpoint_path, amp=None, use_cuda=False):
    try:
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'), pickle_module=pickle_tts)
    model.load_state_dict(state['model'])
    if amp and 'amp' in state:
        amp.load_state_dict(state['amp'])
    if use_cuda:
        model.cuda()
    # set model stepsize
    if hasattr(model.decoder, 'r'):
        model.decoder.set_r(state['r'])
        print(" > Model r: ", state['r'])
    return model, state


def save_model(model, optimizer, current_step, epoch, r, output_path, amp_state_dict=None, **kwargs):
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    state = {
        'model': model_state,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': r
    }
    if amp_state_dict:
        state['amp'] = amp_state_dict
    state.update(kwargs)
    torch.save(state, output_path)


def save_checkpoint(model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    file_name = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print(" > CHECKPOINT : {}".format(checkpoint_path))
    save_model(model, optimizer, current_step, epoch, r, checkpoint_path, **kwargs)


def save_best_model(target_loss, best_loss, model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    if target_loss < best_loss:
        file_name = 'best_model.pth.tar'
        checkpoint_path = os.path.join(output_folder, file_name)
        print(" >> BEST MODEL : {}".format(checkpoint_path))
        save_model(model, optimizer, current_step, epoch, r, checkpoint_path, model_loss=target_loss, **kwargs)
        best_loss = target_loss
    return best_loss
