import os
import json
import re
import torch
import datetime


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if type(value) == str:
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()


def load_checkpoint(model, checkpoint_path, use_cuda=False):
    state =  torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    if use_cuda:
        model.cuda()
    # set model stepsize
    if 'r' in state.keys():
        model.decoder.set_r(state['r'])
    return model, state


def save_model(model, optimizer, current_step, epoch, r, output_folder, file_name, **kwargs):
    checkpoint_path = os.path.join(output_folder, file_name)

    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': model.decoder.r
    }
    state.update(kwargs)
    torch.save(state, checkpoint_path)


def save_checkpoint(model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    print(" > CHECKPOINT : {}".format(checkpoint_path))
    file_name = 'checkpoint_{}.pth.tar'.format(current_step)
    save_model(model, optimizer, current_step, epoch ,r, output_folder, file_name, **kwargs)


def save_best_model(target_loss, best_loss, model, optimizer, current_step, epoch, r, output_folder, **kwargs):
    if target_loss < best_loss:
        print(" > BEST MODEL : {}".format(checkpoint_path))
        file_name = 'best_model.pth.tar'
        save_model(model, optimizer, current_step, epoch ,r, output_folder, file_name, model_loss=target_loss)
        best_loss = target_loss
    return best_loss