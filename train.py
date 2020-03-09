import argparse
import os
import sys
import glob
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TTS.datasets.TTSDataset import MyDataset
from distribute import (DistributedSampler, apply_gradient_allreduce,
                        init_distributed, reduce_tensor)
from TTS.layers.losses import L1LossMasked, MSELossMasked, BCELossMasked
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import (
    NoamLR, check_update, count_parameters, create_experiment_folder,
    get_git_branch, load_config, remove_experiment_folder, save_best_model,
    save_checkpoint, adam_weight_decay, set_init_dict, copy_config_file,
    setup_model, gradual_training_scheduler, KeepAverage,
    set_weight_decay, check_config)
from TTS.utils.logger import Logger
from TTS.utils.speakers import load_speaker_mapping, save_speaker_mapping, \
    get_speakers
from TTS.utils.synthesis import synthesis
from TTS.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.visual import plot_alignment, plot_spectrogram
from TTS.datasets.preprocess import load_meta_data
from TTS.utils.radam import RAdam
from TTS.utils.measures import alignment_diagonal_score

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap, r, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            r,
            c.text_cleaner,
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
            verbose=verbose)
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


def format_data(data):
    if c.use_speaker_embedding:
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
        speaker_ids = [
            speaker_mapping[speaker_name] for speaker_name in speaker_names
        ]
        speaker_ids = torch.LongTensor(speaker_ids)
    else:
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
    return text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, avg_text_length, avg_spec_length


def train(model, criterion, criterion_st, optimizer, optimizer_st, scheduler,
          ap, global_step, epoch):
    data_loader = setup_loader(ap, model.decoder.r, is_val=False,
                               verbose=(epoch == 0))
    model.train()
    epoch_time = 0
    train_values = {
        'avg_postnet_loss': 0,
        'avg_decoder_loss': 0,
        'avg_stop_loss': 0,
        'avg_align_score': 0,
        'avg_step_time': 0,
        'avg_loader_time': 0,
        'avg_alignment_score': 0
    }
    if c.bidirectional_decoder:
        train_values['avg_decoder_b_loss'] = 0  # decoder backward loss
        train_values['avg_decoder_c_loss'] = 0  # decoder consistency loss
    keep_avg = KeepAverage()
    keep_avg.add_values(train_values)
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, avg_text_length, avg_spec_length = format_data(data)
        loader_time = time.time() - end_time

        global_step += 1

        # setup lr
        if c.noam_schedule:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st:
            optimizer_st.zero_grad()

        # forward pass model
        if c.bidirectional_decoder:
            decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
        else:
            decoder_output, postnet_output, alignments, stop_tokens = model(
                text_input, text_lengths, mel_input, speaker_ids=speaker_ids)

        # loss computation
        stop_loss = criterion_st(stop_tokens,
                                 stop_targets, mel_lengths) if c.stopnet else torch.zeros(1)
        if c.loss_masking:
            decoder_loss = criterion(decoder_output, mel_input, mel_lengths)
            if c.model in ["Tacotron", "TacotronGST"]:
                postnet_loss = criterion(postnet_output, linear_input,
                                         mel_lengths)
            else:
                postnet_loss = criterion(postnet_output, mel_input,
                                         mel_lengths)
        else:
            decoder_loss = criterion(decoder_output, mel_input)
            if c.model in ["Tacotron", "TacotronGST"]:
                postnet_loss = criterion(postnet_output, linear_input)
            else:
                postnet_loss = criterion(postnet_output, mel_input)
        loss = decoder_loss + postnet_loss
        if not c.separate_stopnet and c.stopnet:
            loss += stop_loss

        # backward decoder
        if c.bidirectional_decoder:
            if c.loss_masking:
                decoder_backward_loss = criterion(torch.flip(decoder_backward_output, dims=(1, )), mel_input, mel_lengths)
            else:
                decoder_backward_loss = criterion(torch.flip(decoder_backward_output, dims=(1, )), mel_input)
            decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_backward_output, dims=(1, )), decoder_output)
            loss += decoder_backward_loss + decoder_c_loss
            keep_avg.update_values({'avg_decoder_b_loss': decoder_backward_loss.item(), 'avg_decoder_c_loss': decoder_c_loss.item()})

        loss.backward()
        optimizer, current_lr = adam_weight_decay(optimizer)
        grad_norm, grad_flag = check_update(model, c.grad_clip, ignore_stopnet=True)
        optimizer.step()

        # compute alignment score
        align_score = alignment_diagonal_score(alignments)
        keep_avg.update_value('avg_align_score', align_score)

        # backpass and check the grad norm for stop loss
        if c.separate_stopnet:
            stop_loss.backward()
            optimizer_st, _ = adam_weight_decay(optimizer_st)
            grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0)
            optimizer_st.step()
        else:
            grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        if global_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  PostnetLoss:{:.5f}  "
                "DecoderLoss:{:.5f}  StopLoss:{:.5f}  AlignScore:{:.4f}  GradNorm:{:.5f}  "
                "GradNormST:{:.5f}  AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  StepTime:{:.2f}  "
                "LoaderTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, global_step, postnet_loss.item(),
                    decoder_loss.item(), stop_loss.item(), align_score,
                    grad_norm, grad_norm_st, avg_text_length, avg_spec_length,
                    step_time, loader_time, current_lr),
                flush=True)

        # aggregate losses from processes
        if num_gpus > 1:
            postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
            decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
            loss = reduce_tensor(loss.data, num_gpus)
            stop_loss = reduce_tensor(stop_loss.data,
                                      num_gpus) if c.stopnet else stop_loss

        if args.rank == 0:
            update_train_values = {
                'avg_postnet_loss':
                float(postnet_loss.item()),
                'avg_decoder_loss':
                float(decoder_loss.item()),
                'avg_stop_loss':
                stop_loss
                if isinstance(stop_loss, float) else float(stop_loss.item()),
                'avg_step_time':
                step_time,
                'avg_loader_time':
                loader_time
            }
            keep_avg.update_values(update_train_values)

            # Plot Training Iter Stats
            # reduce TB load
            if global_step % 10 == 0:
                iter_stats = {
                    "loss_posnet": postnet_loss.item(),
                    "loss_decoder": decoder_loss.item(),
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "grad_norm_st": grad_norm_st,
                    "step_time": step_time
                }
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, optimizer_st,
                                    postnet_loss.item(), OUT_PATH, global_step,
                                    epoch)

                # Diagnostic visualizations
                const_spec = postnet_output[0].data.cpu().numpy()
                gt_spec = linear_input[0].data.cpu().numpy() if c.model in [
                    "Tacotron", "TacotronGST"
                ] else mel_input[0].data.cpu().numpy()
                align_img = alignments[0].data.cpu().numpy()

                figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img),
                }

                if c.bidirectional_decoder:
                    figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy())

                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_mel_spectrogram(const_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()

    # print epoch stats
    print("   | > EPOCH END -- GlobalStep:{}  "
          "AvgPostnetLoss:{:.5f}  AvgDecoderLoss:{:.5f}  "
          "AvgStopLoss:{:.5f}  AvgAlignScore:{:3f}  EpochTime:{:.2f}  "
          "AvgStepTime:{:.2f}  AvgLoaderTime:{:.2f}".format(
              global_step, keep_avg['avg_postnet_loss'],
              keep_avg['avg_decoder_loss'], keep_avg['avg_stop_loss'],
              keep_avg['avg_align_score'], epoch_time,
              keep_avg['avg_step_time'], keep_avg['avg_loader_time']),
          flush=True)
    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {
            "loss_postnet": keep_avg['avg_postnet_loss'],
            "loss_decoder": keep_avg['avg_decoder_loss'],
            "stop_loss": keep_avg['avg_stop_loss'],
            "alignment_score": keep_avg['avg_align_score'],
            "epoch_time": epoch_time
        }
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return keep_avg['avg_postnet_loss'], global_step


@torch.no_grad()
def evaluate(model, criterion, criterion_st, ap, global_step, epoch):
    data_loader = setup_loader(ap, model.decoder.r, is_val=True)
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.eval()
    epoch_time = 0
    eval_values_dict = {
        'avg_postnet_loss': 0,
        'avg_decoder_loss': 0,
        'avg_stop_loss': 0,
        'avg_align_score': 0
    }
    if c.bidirectional_decoder:
        eval_values_dict['avg_decoder_b_loss'] = 0  # decoder backward loss
        eval_values_dict['avg_decoder_c_loss'] = 0  # decoder consistency loss
    keep_avg = KeepAverage()
    keep_avg.add_values(eval_values_dict)
    print("\n > Validation")

    if data_loader is not None:
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # format data
            text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, _, _ = format_data(data)
            assert mel_input.shape[1] % model.decoder.r == 0

            # forward pass model
            if c.bidirectional_decoder:
                decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                    text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
            else:
                decoder_output, postnet_output, alignments, stop_tokens = model(
                    text_input, text_lengths, mel_input, speaker_ids=speaker_ids)

            # loss computation
            stop_loss = criterion_st(
                stop_tokens, stop_targets, mel_lengths) if c.stopnet else torch.zeros(1)
            if c.loss_masking:
                decoder_loss = criterion(decoder_output, mel_input,
                                         mel_lengths)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(postnet_output, linear_input,
                                             mel_lengths)
                else:
                    postnet_loss = criterion(postnet_output, mel_input,
                                             mel_lengths)
            else:
                decoder_loss = criterion(decoder_output, mel_input)
                if c.model in ["Tacotron", "TacotronGST"]:
                    postnet_loss = criterion(postnet_output, linear_input)
                else:
                    postnet_loss = criterion(postnet_output, mel_input)
            loss = decoder_loss + postnet_loss + stop_loss

            # backward decoder loss
            if c.bidirectional_decoder:
                if c.loss_masking:
                    decoder_backward_loss = criterion(torch.flip(decoder_backward_output, dims=(1, )), mel_input, mel_lengths)
                else:
                    decoder_backward_loss = criterion(torch.flip(decoder_backward_output, dims=(1, )), mel_input)
                decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_backward_output, dims=(1, )), decoder_output)
                loss += decoder_backward_loss + decoder_c_loss
                keep_avg.update_values({'avg_decoder_b_loss': decoder_backward_loss.item(), 'avg_decoder_c_loss': decoder_c_loss.item()})

            step_time = time.time() - start_time
            epoch_time += step_time

            # compute alignment score
            align_score = alignment_diagonal_score(alignments)
            keep_avg.update_value('avg_align_score', align_score)

            # aggregate losses from processes
            if num_gpus > 1:
                postnet_loss = reduce_tensor(postnet_loss.data, num_gpus)
                decoder_loss = reduce_tensor(decoder_loss.data, num_gpus)
                if c.stopnet:
                    stop_loss = reduce_tensor(stop_loss.data, num_gpus)

            keep_avg.update_values({
                'avg_postnet_loss':
                float(postnet_loss.item()),
                'avg_decoder_loss':
                float(decoder_loss.item()),
                'avg_stop_loss':
                float(stop_loss.item()),
            })

            if num_iter % c.print_step == 0:
                print(
                    "   | > TotalLoss: {:.5f}   PostnetLoss: {:.5f} - {:.5f}  DecoderLoss:{:.5f} - {:.5f} "
                    "StopLoss: {:.5f} - {:.5f}  AlignScore: {:.4f} : {:.4f}"
                    .format(loss.item(), postnet_loss.item(),
                            keep_avg['avg_postnet_loss'],
                            decoder_loss.item(),
                            keep_avg['avg_decoder_loss'], stop_loss.item(),
                            keep_avg['avg_stop_loss'], align_score,
                            keep_avg['avg_align_score']),
                    flush=True)

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            const_spec = postnet_output[idx].data.cpu().numpy()
            gt_spec = linear_input[idx].data.cpu().numpy() if c.model in [
                "Tacotron", "TacotronGST"
            ] else mel_input[idx].data.cpu().numpy()
            align_img = alignments[idx].data.cpu().numpy()

            eval_figures = {
                "prediction": plot_spectrogram(const_spec, ap),
                "ground_truth": plot_spectrogram(gt_spec, ap),
                "alignment": plot_alignment(align_img)
            }

            # Sample audio
            if c.model in ["Tacotron", "TacotronGST"]:
                eval_audio = ap.inv_spectrogram(const_spec.T)
            else:
                eval_audio = ap.inv_mel_spectrogram(const_spec.T)
            tb_logger.tb_eval_audios(global_step, {"ValAudio": eval_audio},
                                     c.audio["sample_rate"])

            # Plot Validation Stats
            epoch_stats = {
                "loss_postnet": keep_avg['avg_postnet_loss'],
                "loss_decoder": keep_avg['avg_decoder_loss'],
                "stop_loss": keep_avg['avg_stop_loss'],
                "alignment_score": keep_avg['avg_align_score']
            }

            if c.bidirectional_decoder:
                epoch_stats['loss_decoder_backward'] = keep_avg['avg_decoder_b_loss']
                align_b_img = alignments_backward[idx].data.cpu().numpy()
                eval_figures['alignment_backward'] = plot_alignment(align_b_img)
            tb_logger.tb_eval_stats(global_step, epoch_stats)
            tb_logger.tb_eval_figures(global_step, eval_figures)

    if args.rank == 0 and epoch > c.test_delay_epochs:
        if c.test_sentences_file is None:
            test_sentences = [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "Be a voice, not an echo.",
                "I'm sorry Dave. I'm afraid I can't do that.",
                "This cake is great. It's so delicious and moist."
            ]
        else:
            with open(c.test_sentences_file, "r") as f:
                test_sentences = [s.strip() for s in f.readlines()]

        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        speaker_id = 0 if c.use_speaker_embedding else None
        style_wav = c.get("style_wav_for_test")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, decoder_output, postnet_output, stop_tokens = synthesis(
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
                    postnet_output, ap)
                test_figures['{}-alignment'.format(idx)] = plot_alignment(
                    alignment)
            except:
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(global_step, test_audios,
                                 c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return keep_avg['avg_postnet_loss']


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

    # parse speakers
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model = setup_model(num_chars, num_speakers, c)

    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    params = set_weight_decay(model, c.wd)
    optimizer = RAdam(params, lr=c.lr, weight_decay=0)
    if c.stopnet and c.separate_stopnet:
        optimizer_st = RAdam(model.decoder.stopnet.parameters(),
                             lr=c.lr,
                             weight_decay=0)
    else:
        optimizer_st = None

    if c.loss_masking:
        criterion = L1LossMasked(c.seq_len_norm) if c.model in ["Tacotron", "TacotronGST"
                                                  ] else MSELossMasked(c.seq_len_norm)
    else:
        criterion = nn.L1Loss() if c.model in ["Tacotron", "TacotronGST"
                                               ] else nn.MSELoss()
    criterion_st = BCELossMasked(
        pos_weight=torch.tensor(10)) if c.stopnet else None

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
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
        if criterion_st:
            criterion_st.cuda()

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
        # set gradual training
        if c.gradual_training is not None:
            r, c.batch_size = gradual_training_scheduler(global_step, c)
            c.r = r
            model.decoder.set_r(r)
            if c.bidirectional_decoder:
                model.decoder_backward.set_r(r)
        print(" > Number of outputs per iteration:", model.decoder.r)

        train_loss, global_step = train(model, criterion, criterion_st,
                                        optimizer, optimizer_st, scheduler, ap,
                                        global_step, epoch)
        val_loss = evaluate(model, criterion, criterion_st, ap, global_step,
                            epoch)
        print(" | > Training Loss: {:.5f}   Validation Loss: {:.5f}".format(
            train_loss, val_loss),
              flush=True)
        target_loss = train_loss
        if c.run_eval:
            target_loss = val_loss
        best_loss = save_best_model(model, optimizer, target_loss, best_loss,
                                    OUT_PATH, global_step, epoch)


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
                        default=True,
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

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

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

    if args.rank == 0:
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

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
