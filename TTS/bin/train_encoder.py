#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader
from trainer.io import copy_model_files, save_best_model, save_checkpoint
from trainer.torch import NoamLR
from trainer.trainer_utils import get_optimizer

from TTS.encoder.dataset import EncoderDataset
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.encoder.utils.training import init_training
from TTS.encoder.utils.visual import plot_embeddings
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import count_parameters, remove_experiment_folder
from TTS.utils.samplers import PerfectBatchSampler
from TTS.utils.training import check_update

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap: AudioProcessor, is_val: bool = False, verbose: bool = False):
    num_utter_per_class = c.num_utter_per_class if not is_val else c.eval_num_utter_per_class
    num_classes_in_batch = c.num_classes_in_batch if not is_val else c.eval_num_classes_in_batch

    dataset = EncoderDataset(
        c,
        ap,
        meta_data_eval if is_val else meta_data_train,
        voice_len=c.voice_len,
        num_utter_per_class=num_utter_per_class,
        num_classes_in_batch=num_classes_in_batch,
        verbose=verbose,
        augmentation_config=c.audio_augmentation if not is_val else None,
        use_torch_spec=c.model_params.get("use_torch_spec", False),
    )
    # get classes list
    classes = dataset.get_class_list()

    sampler = PerfectBatchSampler(
        dataset.items,
        classes,
        batch_size=num_classes_in_batch * num_utter_per_class,  # total batch size
        num_classes_in_batch=num_classes_in_batch,
        num_gpus=1,
        shuffle=not is_val,
        drop_last=True,
    )

    if len(classes) < num_classes_in_batch:
        if is_val:
            raise RuntimeError(
                f"config.eval_num_classes_in_batch ({num_classes_in_batch}) need to be <= {len(classes)} (Number total of Classes in the Eval dataset) !"
            )
        raise RuntimeError(
            f"config.num_classes_in_batch ({num_classes_in_batch}) need to be <= {len(classes)} (Number total of Classes in the Train dataset) !"
        )

    # set the classes to avoid get wrong class_id when the number of training and eval classes are not equal
    if is_val:
        dataset.set_classes(train_classes)

    loader = DataLoader(
        dataset,
        num_workers=c.num_loader_workers,
        batch_sampler=sampler,
        collate_fn=dataset.collate_fn,
    )

    return loader, classes, dataset.get_map_classid_to_classname()


def evaluation(model, criterion, data_loader, global_step):
    eval_loss = 0
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            # setup input data
            inputs, labels = data

            # agroup samples of each class in the batch. perfect sampler produces [3,2,1,3,2,1] we need [3,3,2,2,1,1]
            labels = torch.transpose(
                labels.view(c.eval_num_utter_per_class, c.eval_num_classes_in_batch), 0, 1
            ).reshape(labels.shape)
            inputs = torch.transpose(
                inputs.view(c.eval_num_utter_per_class, c.eval_num_classes_in_batch, -1), 0, 1
            ).reshape(inputs.shape)

            # dispatch data to GPU
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # forward pass model
            outputs = model(inputs)

            # loss computation
            loss = criterion(
                outputs.view(c.eval_num_classes_in_batch, outputs.shape[0] // c.eval_num_classes_in_batch, -1), labels
            )

            eval_loss += loss.item()

    eval_avg_loss = eval_loss / len(data_loader)
    # save stats
    dashboard_logger.eval_stats(global_step, {"loss": eval_avg_loss})
    # plot the last batch in the evaluation
    figures = {
        "UMAP Plot": plot_embeddings(outputs.detach().cpu().numpy(), c.num_classes_in_batch),
    }
    dashboard_logger.eval_figures(global_step, figures)
    return eval_avg_loss


def train(model, optimizer, scheduler, criterion, data_loader, eval_data_loader, global_step):
    model.train()
    best_loss = {"train_loss": None, "eval_loss": float("inf")}
    avg_loader_time = 0
    end_time = time.time()
    for epoch in range(c.epochs):
        tot_loss = 0
        epoch_time = 0
        for _, data in enumerate(data_loader):
            start_time = time.time()

            # setup input data
            inputs, labels = data
            # agroup samples of each class in the batch. perfect sampler produces [3,2,1,3,2,1] we need [3,3,2,2,1,1]
            labels = torch.transpose(labels.view(c.num_utter_per_class, c.num_classes_in_batch), 0, 1).reshape(
                labels.shape
            )
            inputs = torch.transpose(inputs.view(c.num_utter_per_class, c.num_classes_in_batch, -1), 0, 1).reshape(
                inputs.shape
            )
            # ToDo: move it to a unit test
            # labels_converted = torch.transpose(labels.view(c.num_utter_per_class, c.num_classes_in_batch), 0, 1).reshape(labels.shape)
            # inputs_converted = torch.transpose(inputs.view(c.num_utter_per_class, c.num_classes_in_batch, -1), 0, 1).reshape(inputs.shape)
            # idx = 0
            # for j in range(0, c.num_classes_in_batch, 1):
            #     for i in range(j, len(labels), c.num_classes_in_batch):
            #         if not torch.all(labels[i].eq(labels_converted[idx])) or not torch.all(inputs[i].eq(inputs_converted[idx])):
            #             print("Invalid")
            #             print(labels)
            #             exit()
            #         idx += 1
            # labels = labels_converted
            # inputs = inputs_converted

            loader_time = time.time() - end_time
            global_step += 1

            # setup lr
            if c.lr_decay:
                scheduler.step()
            optimizer.zero_grad()

            # dispatch data to GPU
            if use_cuda:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            # forward pass model
            outputs = model(inputs)

            # loss computation
            loss = criterion(
                outputs.view(c.num_classes_in_batch, outputs.shape[0] // c.num_classes_in_batch, -1), labels
            )
            loss.backward()
            grad_norm, _ = check_update(model, c.grad_clip)
            optimizer.step()

            step_time = time.time() - start_time
            epoch_time += step_time

            # acumulate the total epoch loss
            tot_loss += loss.item()

            # Averaged Loader Time
            num_loader_workers = c.num_loader_workers if c.num_loader_workers > 0 else 1
            avg_loader_time = (
                1 / num_loader_workers * loader_time + (num_loader_workers - 1) / num_loader_workers * avg_loader_time
                if avg_loader_time != 0
                else loader_time
            )
            current_lr = optimizer.param_groups[0]["lr"]

            if global_step % c.steps_plot_stats == 0:
                # Plot Training Epoch Stats
                train_stats = {
                    "loss": loss.item(),
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                    "avg_loader_time": avg_loader_time,
                }
                dashboard_logger.train_epoch_stats(global_step, train_stats)
                figures = {
                    "UMAP Plot": plot_embeddings(outputs.detach().cpu().numpy(), c.num_classes_in_batch),
                }
                dashboard_logger.train_figures(global_step, figures)

            if global_step % c.print_step == 0:
                print(
                    "   | > Step:{}  Loss:{:.5f}  GradNorm:{:.5f}  "
                    "StepTime:{:.2f}  LoaderTime:{:.2f}  AvGLoaderTime:{:.2f}  LR:{:.6f}".format(
                        global_step, loss.item(), grad_norm, step_time, loader_time, avg_loader_time, current_lr
                    ),
                    flush=True,
                )

            if global_step % c.save_step == 0:
                # save model
                save_checkpoint(
                    c, model, optimizer, None, global_step, epoch, OUT_PATH, criterion=criterion.state_dict()
                )

            end_time = time.time()

        print("")
        print(
            ">>> Epoch:{}  AvgLoss: {:.5f} GradNorm:{:.5f}  "
            "EpochTime:{:.2f} AvGLoaderTime:{:.2f} ".format(
                epoch, tot_loss / len(data_loader), grad_norm, epoch_time, avg_loader_time
            ),
            flush=True,
        )
        # evaluation
        if c.run_eval:
            model.eval()
            eval_loss = evaluation(model, criterion, eval_data_loader, global_step)
            print("\n\n")
            print("--> EVAL PERFORMANCE")
            print(
                "   | > Epoch:{}  AvgLoss: {:.5f} ".format(epoch, eval_loss),
                flush=True,
            )
            # save the best checkpoint
            best_loss = save_best_model(
                {"train_loss": None, "eval_loss": eval_loss},
                best_loss,
                c,
                model,
                optimizer,
                None,
                global_step,
                epoch,
                OUT_PATH,
                criterion=criterion.state_dict(),
            )
            model.train()

    return best_loss, global_step


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train
    global meta_data_eval
    global train_classes

    ap = AudioProcessor(**c.audio)
    model = setup_encoder_model(c)

    optimizer = get_optimizer(c.optimizer, c.optimizer_params, c.lr, model)

    # pylint: disable=redefined-outer-name
    meta_data_train, meta_data_eval = load_tts_samples(c.datasets, eval_split=True)

    train_data_loader, train_classes, map_classid_to_classname = setup_loader(ap, is_val=False, verbose=True)
    if c.run_eval:
        eval_data_loader, _, _ = setup_loader(ap, is_val=True, verbose=True)
    else:
        eval_data_loader = None

    num_classes = len(train_classes)
    criterion = model.get_criterion(c, num_classes)

    if c.loss == "softmaxproto" and c.model != "speaker_encoder":
        c.map_classid_to_classname = map_classid_to_classname
        copy_model_files(c, OUT_PATH, new_fields={})

    if args.restore_path:
        criterion, args.restore_step = model.load_checkpoint(
            c, args.restore_path, eval=False, use_cuda=use_cuda, criterion=criterion
        )
        print(" > Model restored from step %d" % args.restore_step, flush=True)
    else:
        args.restore_step = 0

    if c.lr_decay:
        scheduler = NoamLR(optimizer, warmup_steps=c.warmup_steps, last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if use_cuda:
        model = model.cuda()
        criterion.cuda()

    global_step = args.restore_step
    _, global_step = train(model, optimizer, scheduler, criterion, train_data_loader, eval_data_loader, global_step)


if __name__ == "__main__":
    args, c, OUT_PATH, AUDIO_PATH, c_logger, dashboard_logger = init_training()

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
