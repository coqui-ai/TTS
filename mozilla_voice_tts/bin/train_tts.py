#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from mozilla_voice_tts.tts.datasets.preprocess import load_meta_data
from mozilla_voice_tts.tts.datasets.TTSDataset import MyDataset
from mozilla_voice_tts.tts.layers.losses import TacotronLoss
from mozilla_voice_tts.tts.utils.distribute import (DistributedSampler,
                                                    apply_gradient_allreduce,
                                                    init_distributed,
                                                    reduce_tensor)
from mozilla_voice_tts.tts.utils.generic_utils import check_config, setup_model
from mozilla_voice_tts.tts.utils.io import save_best_model, save_checkpoint
from mozilla_voice_tts.tts.utils.measures import alignment_diagonal_score
from mozilla_voice_tts.tts.utils.speakers import (get_speakers,
                                                  load_speaker_mapping,
                                                  save_speaker_mapping)
from mozilla_voice_tts.tts.utils.synthesis import synthesis
from mozilla_voice_tts.tts.utils.text.symbols import (make_symbols, phonemes,
                                                      symbols)
from mozilla_voice_tts.tts.utils.visual import plot_alignment, plot_spectrogram
from mozilla_voice_tts.utils.audio import AudioProcessor
from mozilla_voice_tts.utils.console_logger import ConsoleLogger
from mozilla_voice_tts.utils.generic_utils import (KeepAverage,
                                                   count_parameters,
                                                   create_experiment_folder,
                                                   get_git_branch,
                                                   remove_experiment_folder,
                                                   set_init_dict)
from mozilla_voice_tts.utils.io import copy_config_file, load_config
from mozilla_voice_tts.utils.radam import RAdam
from mozilla_voice_tts.utils.tensorboard_logger import TensorboardLogger
from mozilla_voice_tts.utils.training import (NoamLR, adam_weight_decay,
                                              check_update,
                                              gradual_training_scheduler,
                                              set_weight_decay,
                                              setup_torch_training_env)

use_cuda, num_gpus = setup_torch_training_env(True, False)


def setup_loader(ap, r, is_val=False, verbose=False, speaker_mapping=None):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            r,
            c.text_cleaner,
            compute_linear_spec=c.model.lower() == 'tacotron',
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            tp=c.characters if 'characters' in c.keys() else None,
            batch_group_size=0 if is_val else c.batch_group_size *
            c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose,
            speaker_mapping=speaker_mapping if c.use_speaker_embedding and c.use_external_speaker_embedding_file else None)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader

def format_data(data, speaker_mapping=None):
    if speaker_mapping is None and c.use_speaker_embedding and not c.use_external_speaker_embedding_file:
        speaker_mapping = load_speaker_mapping(OUT_PATH)

    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    linear_input = data[3] if c.model in ["Tacotron"] else None
    mel_input = data[4]
    mel_lengths = data[5]
    stop_targets = data[6]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if c.use_speaker_embedding:
        if c.use_external_speaker_embedding_file:
            speaker_embeddings = data[8]
            speaker_ids = None
        else:
            speaker_ids = [
                speaker_mapping[speaker_name] for speaker_name in speaker_names
            ]
            speaker_ids = torch.LongTensor(speaker_ids)
            speaker_embeddings = None
    else:
        speaker_embeddings = None
        speaker_ids = None


    # set stop targets view, we predict a single stop token per iteration.
    stop_targets = stop_targets.view(text_input.shape[0],
                                     stop_targets.size(1) // c.r, -1)
    stop_targets = (stop_targets.sum(2) >
                    0.0).unsqueeze(2).float().squeeze(2)

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        linear_input = linear_input.cuda(non_blocking=True) if c.model in ["Tacotron"] else None
        stop_targets = stop_targets.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.cuda(non_blocking=True)

    return text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, speaker_embeddings, avg_text_length, avg_spec_length


def train(model, criterion, optimizer, optimizer_st, scheduler,
          ap, global_step, epoch, amp, speaker_mapping=None):
    data_loader = setup_loader(ap, model.decoder.r, is_val=False,
                               verbose=(epoch == 0), speaker_mapping=speaker_mapping)
    model.train()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, speaker_embeddings, avg_text_length, avg_spec_length = format_data(data, speaker_mapping)
        loader_time = time.time() - end_time

        global_step += 1

        # setup lr
        if c.noam_schedule:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st:
            optimizer_st.zero_grad()

        # forward pass model
        if c.bidirectional_decoder or c.double_decoder_consistency:
            decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                text_input, text_lengths, mel_input, mel_lengths, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
        else:
            decoder_output, postnet_output, alignments, stop_tokens = model(
                text_input, text_lengths, mel_input, mel_lengths, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
            decoder_backward_output = None
            alignments_backward = None

        # set the [alignment] lengths wrt reduction factor for guided attention
        if mel_lengths.max() % model.decoder.r != 0:
            alignment_lengths = (mel_lengths + (model.decoder.r - (mel_lengths.max() % model.decoder.r))) // model.decoder.r
        else:
            alignment_lengths = mel_lengths //  model.decoder.r

        # compute loss
        loss_dict = criterion(postnet_output, decoder_output, mel_input,
                              linear_input, stop_tokens, stop_targets,
                              mel_lengths, decoder_backward_output,
                              alignments, alignment_lengths, alignments_backward,
                              text_lengths)

        # backward pass
        if amp is not None:
            with amp.scale_loss(loss_dict['loss'], optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_dict['loss'].backward()

        optimizer, current_lr = adam_weight_decay(optimizer)
        if amp:
            amp_opt_params = amp.master_params(optimizer)
        else:
            amp_opt_params = None
        grad_norm, _ = check_update(model, c.grad_clip, ignore_stopnet=True, amp_opt_params=amp_opt_params)
        optimizer.step()

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(alignments)
        loss_dict['align_error'] = align_error

        # backpass and check the grad norm for stop loss
        if c.separate_stopnet:
            loss_dict['stopnet_loss'].backward()
            optimizer_st, _ = adam_weight_decay(optimizer_st)
            if amp:
                amp_opt_params = amp.master_params(optimizer)
            else:
                amp_opt_params = None
            grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0, amp_opt_params=amp_opt_params)
            optimizer_st.step()
        else:
            grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        # aggregate losses from processes
        if num_gpus > 1:
            loss_dict['postnet_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
            loss_dict['decoder_loss'] = reduce_tensor(loss_dict['decoder_loss'].data, num_gpus)
            loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)
            loss_dict['stopnet_loss'] = reduce_tensor(loss_dict['stopnet_loss'].data, num_gpus) if c.stopnet else loss_dict['stopnet_loss']

        # detach loss values
        loss_dict_new = dict()
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_new[key] = value
            else:
                loss_dict_new[key] = value.item()
        loss_dict = loss_dict_new

        # update avg stats
        update_train_values = dict()
        for key, value in loss_dict.items():
            update_train_values['avg_' + key] = value
        update_train_values['avg_loader_time'] = loader_time
        update_train_values['avg_step_time'] = step_time
        keep_avg.update_values(update_train_values)

        # print training progress
        if global_step % c.print_step == 0:
            log_dict = {
                "avg_spec_length": [avg_spec_length, 1],  # value, precision
                "avg_text_length": [avg_text_length, 1],
                "step_time": [step_time, 4],
                "loader_time": [loader_time, 2],
                "current_lr": current_lr,
            }
            c_logger.print_train_step(batch_n_iter, num_iter, global_step,
                                      log_dict, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load
            if global_step % c.tb_plot_step == 0:
                iter_stats = {
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "grad_norm_st": grad_norm_st,
                    "step_time": step_time
                }
                iter_stats.update(loss_dict)
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, global_step, epoch, model.decoder.r, OUT_PATH,
                                    optimizer_st=optimizer_st,
                                    model_loss=loss_dict['postnet_loss'],
                                    amp_state_dict=amp.state_dict() if amp else None)

                # Diagnostic visualizations
                const_spec = postnet_output[0].data.cpu().numpy()
                gt_spec = linear_input[0].data.cpu().numpy() if c.model in [
                    "Tacotron", "TacotronGST"
                ] else mel_input[0].data.cpu().numpy()
                align_img = alignments[0].data.cpu().numpy()

                figures = {
                    "prediction": plot_spectrogram(const_spec, ap, output_fig=False),
                    "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
                    "alignment": plot_alignment(align_img, output_fig=False),
                }

                if c.bidirectional_decoder or c.double_decoder_consistency:
                    figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy(), output_fig=False)

                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_melspectrogram(const_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Epoch Stats
    if args.rank == 0:
        epoch_stats = {"epoch_time": epoch_time}
        epoch_stats.update(keep_avg.avg_values)
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(model, criterion, ap, global_step, epoch, speaker_mapping=None):
    data_loader = setup_loader(ap, model.decoder.r, is_val=True, speaker_mapping=speaker_mapping)
    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    c_logger.print_eval_start()
    if data_loader is not None:
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # format data
            text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, speaker_embeddings, _, _ = format_data(data, speaker_mapping)
            assert mel_input.shape[1] % model.decoder.r == 0

            # forward pass model
            if c.bidirectional_decoder or c.double_decoder_consistency:
                decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                    text_input, text_lengths, mel_input, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
            else:
                decoder_output, postnet_output, alignments, stop_tokens = model(
                    text_input, text_lengths, mel_input, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
                decoder_backward_output = None
                alignments_backward = None

            # set the alignment lengths wrt reduction factor for guided attention
            if mel_lengths.max() % model.decoder.r != 0:
                alignment_lengths = (mel_lengths + (model.decoder.r - (mel_lengths.max() % model.decoder.r))) // model.decoder.r
            else:
                alignment_lengths = mel_lengths //  model.decoder.r

            # compute loss
            loss_dict = criterion(postnet_output, decoder_output, mel_input,
                                  linear_input, stop_tokens, stop_targets,
                                  mel_lengths, decoder_backward_output,
                                  alignments, alignment_lengths, alignments_backward,
                                  text_lengths)

            # step time
            step_time = time.time() - start_time
            epoch_time += step_time

            # compute alignment score
            align_error = 1 - alignment_diagonal_score(alignments)
            loss_dict['align_error'] = align_error

            # aggregate losses from processes
            if num_gpus > 1:
                loss_dict['postnet_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
                loss_dict['decoder_loss'] = reduce_tensor(loss_dict['decoder_loss'].data, num_gpus)
                if c.stopnet:
                    loss_dict['stopnet_loss'] = reduce_tensor(loss_dict['stopnet_loss'].data, num_gpus)

            # detach loss values
            loss_dict_new = dict()
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    loss_dict_new[key] = value
                else:
                    loss_dict_new[key] = value.item()
            loss_dict = loss_dict_new

            # update avg stats
            update_train_values = dict()
            for key, value in loss_dict.items():
                update_train_values['avg_' + key] = value
            keep_avg.update_values(update_train_values)

            if c.print_eval:
                c_logger.print_eval_step(num_iter, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            const_spec = postnet_output[idx].data.cpu().numpy()
            gt_spec = linear_input[idx].data.cpu().numpy() if c.model in [
                "Tacotron", "TacotronGST"
            ] else mel_input[idx].data.cpu().numpy()
            align_img = alignments[idx].data.cpu().numpy()

            eval_figures = {
                "prediction": plot_spectrogram(const_spec, ap, output_fig=False),
                "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
                "alignment": plot_alignment(align_img, output_fig=False)
            }

            # Sample audio
            if c.model in ["Tacotron", "TacotronGST"]:
                eval_audio = ap.inv_spectrogram(const_spec.T)
            else:
                eval_audio = ap.inv_melspectrogram(const_spec.T)
            tb_logger.tb_eval_audios(global_step, {"ValAudio": eval_audio},
                                     c.audio["sample_rate"])

            # Plot Validation Stats

            if c.bidirectional_decoder or c.double_decoder_consistency:
                align_b_img = alignments_backward[idx].data.cpu().numpy()
                eval_figures['alignment2'] = plot_alignment(align_b_img, output_fig=False)
            tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)
            tb_logger.tb_eval_figures(global_step, eval_figures)

    if args.rank == 0 and epoch > c.test_delay_epochs:
        if c.test_sentences_file is None:
            test_sentences = [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "Be a voice, not an echo.",
                "I'm sorry Dave. I'm afraid I can't do that.",
                "This cake is great. It's so delicious and moist.",
                "Prior to November 22, 1963."
            ]
        else:
            with open(c.test_sentences_file, "r") as f:
                test_sentences = [s.strip() for s in f.readlines()]

        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        speaker_id = 0 if c.use_speaker_embedding else None
        style_wav = c.get("gst_style_input")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, decoder_output, postnet_output, stop_tokens, _ = synthesis(
                    model,
                    test_sentence,
                    c,
                    use_cuda,
                    ap,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    truncated=False,
                    enable_eos_bos_chars=c.enable_eos_bos_chars, #pylint: disable=unused-argument
                    use_griffin_lim=True,
                    do_trim_silence=False)

                file_path = os.path.join(AUDIO_PATH, str(global_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)] = plot_spectrogram(
                    postnet_output, ap, output_fig=False)
                test_figures['{}-alignment'.format(idx)] = plot_alignment(
                    alignment, output_fig=False)
            except:  #pylint: disable=bare-except
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(global_step, test_audios,
                                 c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return keep_avg.avg_values


# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train, meta_data_eval, symbols, phonemes
    # Audio processor
    ap = AudioProcessor(**c.audio)
    if 'characters' in c.keys():
        symbols, phonemes = make_symbols(**c.characters)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    # set the portion of the data used for training
    if 'train_portion' in c.keys():
        meta_data_train = meta_data_train[:int(len(meta_data_train) * c.train_portion)]
    if 'eval_portion' in c.keys():
        meta_data_eval = meta_data_eval[:int(len(meta_data_eval) * c.eval_portion)]

    # parse speakers
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            if c.use_external_speaker_embedding_file: # if restore checkpoint and use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                if not speaker_mapping:
                    print("WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file")
                    speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
                    if not speaker_mapping:
                        raise RuntimeError("You must copy the file speakers.json to restore_path, or set a valid file in CONFIG.external_speaker_embedding_file")
                speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
            elif not c.use_external_speaker_embedding_file: # if restore checkpoint and don't use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                speaker_embedding_dim = None
                assert all([speaker in speaker_mapping
                            for speaker in speakers]), "As of now you, you cannot " \
                                                    "introduce new speakers to " \
                                                    "a previously trained model."
        elif c.use_external_speaker_embedding_file and c.external_speaker_embedding_file: # if start new train using External Embedding file
            speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
            speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
        elif c.use_external_speaker_embedding_file and not c.external_speaker_embedding_file: # if start new train using External Embedding file and don't pass external embedding file
            raise "use_external_speaker_embedding_file is True, so you need pass a external speaker embedding file, run GE2E-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb or AngularPrototypical-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb notebook in notebooks/ folder"
        else: # if start new train and don't use External Embedding file
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
            speaker_embedding_dim = None
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0
        speaker_embedding_dim = None
        speaker_mapping = None

    model = setup_model(num_chars, num_speakers, c, speaker_embedding_dim)

    params = set_weight_decay(model, c.wd)
    optimizer = RAdam(params, lr=c.lr, weight_decay=0)
    if c.stopnet and c.separate_stopnet:
        optimizer_st = RAdam(model.decoder.stopnet.parameters(),
                             lr=c.lr,
                             weight_decay=0)
    else:
        optimizer_st = None

    if c.apex_amp_level == "O1":
        # pylint: disable=import-outside-toplevel
        from apex import amp
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level=c.apex_amp_level)
    else:
        amp = None

    # setup criterion
    criterion = TacotronLoss(c, stopnet_pos_weight=10.0, ga_sigma=0.4)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except KeyError:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            # torch.save(model_dict, os.path.join(OUT_PATH, 'state_dict.pt'))
            # print("State Dict saved for debug in: ", os.path.join(OUT_PATH, 'state_dict.pt'))
            model.load_state_dict(model_dict)
            del model_dict

        if amp and 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])

        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(" > Model restored from step %d" % checkpoint['step'],
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model.cuda()
        criterion.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.noam_schedule:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        # set gradual training
        if c.gradual_training is not None:
            r, c.batch_size = gradual_training_scheduler(global_step, c)
            c.r = r
            model.decoder.set_r(r)
            if c.bidirectional_decoder:
                model.decoder_backward.set_r(r)
            print("\n > Number of output frames:", model.decoder.r)
        train_avg_loss_dict, global_step = train(model, criterion, optimizer,
                                                 optimizer_st, scheduler, ap,
                                                 global_step, epoch, amp, speaker_mapping)
        eval_avg_loss_dict = evaluate(model, criterion, ap, global_step, epoch, speaker_mapping)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = train_avg_loss_dict['avg_postnet_loss']
        if c.run_eval:
            target_loss = eval_avg_loss_dict['avg_postnet_loss']
        best_loss = save_best_model(target_loss, best_loss, model, optimizer, global_step, epoch, c.r,
                                    OUT_PATH, amp_state_dict=amp.state_dict() if amp else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Do not verify commit integrity to run training.')

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument('--group_id',
                        type=str,
                        default="",
                        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    if args.continue_path != '':
        args.output_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, 'config.json')
        list_of_files = glob.glob(args.continue_path + "/*.pth.tar") # * means all if need specific format then *.csv
        latest_model_file = max(list_of_files, key=os.path.getctime)
        args.restore_path = latest_model_file
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)
    check_config(c)
    _ = os.path.dirname(os.path.realpath(__file__))

    if c.apex_amp_level == 'O1':
        print("   >  apex AMP level: ", c.apex_amp_level)

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    c_logger = ConsoleLogger()

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path,
                         os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

        LOG_DIR = OUT_PATH
        tb_logger = TensorboardLogger(LOG_DIR, model_name='TTS')

        # write model desc to tensorboard
        tb_logger.tb_add_text('model-description', c['run_description'], 0)

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
