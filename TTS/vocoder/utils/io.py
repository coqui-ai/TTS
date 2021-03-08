import os
import glob
import torch
import datetime
import pickle as pickle_tts

from TTS.utils.io import RenamingUnpickler


def load_checkpoint(model, checkpoint_path, use_cuda=False, eval=False):  # pylint: disable=redefined-builtin
    try:
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'), pickle_module=pickle_tts)
    model.load_state_dict(state['model'])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state


def save_model(model, optimizer, scheduler, model_disc, optimizer_disc,
               scheduler_disc, current_step, epoch, output_path, **kwargs):
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    model_disc_state = model_disc.state_dict()\
         if model_disc is not None else None
    optimizer_state = optimizer.state_dict()\
         if optimizer is not None else None
    optimizer_disc_state = optimizer_disc.state_dict()\
        if optimizer_disc is not None else None
    scheduler_state = scheduler.state_dict()\
        if scheduler is not None else None
    scheduler_disc_state = scheduler_disc.state_dict()\
        if scheduler_disc is not None else None
    state = {
        'model': model_state,
        'optimizer': optimizer_state,
        'scheduler': scheduler_state,
        'model_disc': model_disc_state,
        'optimizer_disc': optimizer_disc_state,
        'scheduler_disc': scheduler_disc_state,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    torch.save(state, output_path)


def save_checkpoint(model, optimizer, scheduler, model_disc, optimizer_disc,
                    scheduler_disc, current_step, epoch, output_folder,
                    **kwargs):
    file_name = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print(" > CHECKPOINT : {}".format(checkpoint_path))
    save_model(model, optimizer, scheduler, model_disc, optimizer_disc,
               scheduler_disc, current_step, epoch, checkpoint_path, **kwargs)


def save_best_model(current_loss, best_loss, model, optimizer, scheduler,
                    model_disc, optimizer_disc, scheduler_disc, current_step,
                    epoch, out_path, keep_all_best=False, keep_after=10000,
                    **kwargs):
    if current_loss < best_loss:
        best_model_name = f'best_model_{current_step}.pth.tar'
        checkpoint_path = os.path.join(out_path, best_model_name)
        print(" > BEST MODEL : {}".format(checkpoint_path))
        save_model(model,
                   optimizer,
                   scheduler,
                   model_disc,
                   optimizer_disc,
                   scheduler_disc,
                   current_step,
                   epoch,
                   checkpoint_path,
                   model_loss=current_loss,
                   **kwargs)
        # only delete previous if current is saved successfully
        if not keep_all_best or (current_step < keep_after):
            model_names = glob.glob(
                os.path.join(out_path, 'best_model*.pth.tar'))
            for model_name in model_names:
                if os.path.basename(model_name) == best_model_name:
                    continue
                os.remove(model_name)
        # create symlink to best model for convinience
        link_name = 'best_model.pth.tar'
        link_path = os.path.join(out_path, link_name)
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.remove(link_path)
        os.symlink(best_model_name, os.path.join(out_path, link_name))
        best_loss = current_loss
    return best_loss
