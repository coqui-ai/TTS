import os
import torch
import datetime


def load_checkpoint(model, checkpoint_path, use_cuda=False):
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    if use_cuda:
        model.cuda()
    # set model stepsize
    if 'r' in state.keys():
        model.decoder.set_r(state['r'])
    return model, state


def save_model(model, optimizer, current_step, epoch, r, output_path, **kwargs):
    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': r
    }
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
        print(" > BEST MODEL : {}".format(checkpoint_path))
        save_model(model, optimizer, current_step, epoch, r, checkpoint_path, model_loss=target_loss, **kwargs)
        best_loss = target_loss
    return best_loss
