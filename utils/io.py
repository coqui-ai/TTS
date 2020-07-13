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
    with open(config_path, "r", encoding = "utf-8") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r", encoding = "utf-8").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if isinstance(value, str):
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()


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
