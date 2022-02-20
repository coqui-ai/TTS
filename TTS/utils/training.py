import numpy as np
import torch


def check_update(model, grad_clip, ignore_stopnet=False, amp_opt_params=None):
    r"""Check model gradient against unexpected jumps and failures"""
    skip_flag = False
    if ignore_stopnet:
        if not amp_opt_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for name, param in model.named_parameters() if "stopnet" not in name], grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp_opt_params, grad_clip)
    else:
        if not amp_opt_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp_opt_params, grad_clip)

    # compatibility with different torch versions
    if isinstance(grad_norm, float):
        if np.isinf(grad_norm):
            print(" | > Gradient is INF !!")
            skip_flag = True
    else:
        if torch.isinf(grad_norm):
            print(" | > Gradient is INF !!")
            skip_flag = True
    return grad_norm, skip_flag


def gradual_training_scheduler(global_step, config):
    """Setup the gradual training schedule wrt number
    of active GPUs"""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    new_values = None
    # we set the scheduling wrt num_gpus
    for values in config.gradual_training:
        if global_step * num_gpus >= values[0]:
            new_values = values
    return new_values[1], new_values[2]
