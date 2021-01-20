import os
import torch
import datetime
import pickle as pickle_tts

from TTS.utils.io import RenamingUnpickler



def load_checkpoint(model, checkpoint_path, amp=None, use_cuda=False, eval=False):
    """Load ```TTS.tts.models``` checkpoints.

    Args:
        model (TTS.tts.models): model object to load the weights for.
        checkpoint_path (string): checkpoint file path.
        amp (apex.amp, optional): Apex amp abject to load apex related state vars. Defaults to None.
        use_cuda (bool, optional): load model to GPU if True. Defaults to False.

    Returns:
        [type]: [description]
    """
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
    if eval:
        model.eval()
    return model, state


def save_model(model, optimizer, current_step, epoch, r, output_path, amp_state_dict=None, **kwargs):
    """Save ```TTS.tts.models``` states with extra fields.

    Args:
        model (TTS.tts.models.Model): models object to be saved.
        optimizer (torch.optim.optimizers.Optimizer): model optimizer used for training.
        current_step (int): current number of training steps.
        epoch (int): current number of training epochs.
        r (int): model reduction rate for Tacotron models.
        output_path (str): output path to save the model file.
        amp_state_dict (state_dict, optional): Apex.amp state dict if Apex is enabled. Defaults to None.
    """
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
    """Save model checkpoint, intended for saving checkpoints at training.

    Args:
        model (TTS.tts.models.Model): models object to be saved.
        optimizer (torch.optim.optimizers.Optimizer): model optimizer used for training.
        current_step (int): current number of training steps.
        epoch (int): current number of training epochs.
        r (int): model reduction rate for Tacotron models.
        output_path (str): output path to save the model file.
    """
    file_name = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print(" > CHECKPOINT : {}".format(checkpoint_path))
    save_model(model, optimizer, current_step, epoch, r, checkpoint_path, **kwargs)


def save_best_model(target_loss, best_loss, model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    """Save model checkpoint, intended for saving the best model after each epoch.
    It compares the current model loss with the best loss so far and saves the
    model if the current loss is better.

    Args:
        target_loss (float): current model loss.
        best_loss (float): best loss so far.
        model (TTS.tts.models.Model): models object to be saved.
        optimizer (torch.optim.optimizers.Optimizer): model optimizer used for training.
        current_step (int): current number of training steps.
        epoch (int): current number of training epochs.
        r (int): model reduction rate for Tacotron models.
        output_path (str): output path to save the model file.

    Returns:
        float: updated current best loss.
    """
    if target_loss < best_loss:
        file_name = 'best_model.pth.tar'
        checkpoint_path = os.path.join(output_folder, file_name)
        print(" >> BEST MODEL : {}".format(checkpoint_path))
        save_model(model, optimizer, current_step, epoch, r, checkpoint_path, model_loss=target_loss, **kwargs)
        best_loss = target_loss
    return best_loss
