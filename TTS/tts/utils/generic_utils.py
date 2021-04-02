import re
import torch
import importlib
import numpy as np
from collections import Counter

from TTS.utils.generic_utils import check_argument


def split_dataset(items):
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = min(500, int(len(items) * 0.01))
    assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item[-1] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx][-1]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]

# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    seq_range = torch.arange(max_len,
                             dtype=sequence_length.dtype,
                             device=sequence_length.device)
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)
    text = text.replace('Tts', 'TTS')
    return text


def setup_model(num_chars, num_speakers, c, speaker_embedding_dim=None):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('TTS.tts.models.' + c.model.lower())
    MyModel = getattr(MyModel, to_camel(c.model))
    if c.model.lower() in "tacotron":
        model = MyModel(num_chars=num_chars + getattr(c, "add_blank", False),
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=int(c.audio['fft_size'] / 2 + 1),
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        gst_embedding_dim=c.gst['gst_embedding_dim'],
                        gst_num_heads=c.gst['gst_num_heads'],
                        gst_style_tokens=c.gst['gst_style_tokens'],
                        gst_use_speaker_embedding=c.gst['gst_use_speaker_embedding'],
                        memory_size=c.memory_size,
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r,
                        speaker_embedding_dim=speaker_embedding_dim)
    elif c.model.lower() == "tacotron2":
        model = MyModel(num_chars=num_chars + getattr(c, "add_blank", False),
                        num_speakers=num_speakers,
                        r=c.r,
                        postnet_output_dim=c.audio['num_mels'],
                        decoder_output_dim=c.audio['num_mels'],
                        gst=c.use_gst,
                        gst_embedding_dim=c.gst['gst_embedding_dim'],
                        gst_num_heads=c.gst['gst_num_heads'],
                        gst_style_tokens=c.gst['gst_style_tokens'],
                        gst_use_speaker_embedding=c.gst['gst_use_speaker_embedding'],
                        attn_type=c.attention_type,
                        attn_win=c.windowing,
                        attn_norm=c.attention_norm,
                        prenet_type=c.prenet_type,
                        prenet_dropout=c.prenet_dropout,
                        forward_attn=c.use_forward_attn,
                        trans_agent=c.transition_agent,
                        forward_attn_mask=c.forward_attn_mask,
                        location_attn=c.location_attn,
                        attn_K=c.attention_heads,
                        separate_stopnet=c.separate_stopnet,
                        bidirectional_decoder=c.bidirectional_decoder,
                        double_decoder_consistency=c.double_decoder_consistency,
                        ddc_r=c.ddc_r,
                        speaker_embedding_dim=speaker_embedding_dim)
    elif c.model.lower() == "glow_tts":
        model = MyModel(num_chars=num_chars + getattr(c, "add_blank", False),
                        hidden_channels_enc=c['hidden_channels_encoder'],
                        hidden_channels_dec=c['hidden_channels_decoder'],
                        hidden_channels_dp=c['hidden_channels_duration_predictor'],
                        out_channels=c.audio['num_mels'],
                        encoder_type=c.encoder_type,
                        encoder_params=c.encoder_params,
                        use_encoder_prenet=c["use_encoder_prenet"],
                        num_flow_blocks_dec=12,
                        kernel_size_dec=5,
                        dilation_rate=1,
                        num_block_layers=4,
                        dropout_p_dec=0.05,
                        num_speakers=num_speakers,
                        c_in_channels=0,
                        num_splits=4,
                        num_squeeze=2,
                        sigmoid_scale=False,
                        mean_only=True,
                        external_speaker_embedding_dim=speaker_embedding_dim)
    elif c.model.lower() == "speedy_speech":
        model = MyModel(num_chars=num_chars + getattr(c, "add_blank", False),
                        out_channels=c.audio['num_mels'],
                        hidden_channels=c['hidden_channels'],
                        positional_encoding=c['positional_encoding'],
                        encoder_type=c['encoder_type'],
                        encoder_params=c['encoder_params'],
                        decoder_type=c['decoder_type'],
                        decoder_params=c['decoder_params'],
                        c_in_channels=0)
    elif c.model.lower() == "align_tts":
        model = MyModel(num_chars=num_chars + getattr(c, "add_blank", False),
                        out_channels=c.audio['num_mels'],
                        hidden_channels=c['hidden_channels'],
                        hidden_channels_dp=c['hidden_channels_dp'],
                        encoder_type=c['encoder_type'],
                        encoder_params=c['encoder_params'],
                        decoder_type=c['decoder_type'],
                        decoder_params=c['decoder_params'],
                        c_in_channels=0)
    return model

def is_tacotron(c):
    return 'tacotron' in c['model'].lower()

def check_config_tts(c):
    check_argument('model', c, enum_list=['tacotron', 'tacotron2', 'glow_tts', 'speedy_speech', 'align_tts'], restricted=True, val_type=str)
    check_argument('run_name', c, restricted=True, val_type=str)
    check_argument('run_description', c, val_type=str)

    # AUDIO
    check_argument('audio', c, restricted=True, val_type=dict)

    # audio processing parameters
    check_argument('num_mels', c['audio'], restricted=True, val_type=int, min_val=10, max_val=2056)
    check_argument('fft_size', c['audio'], restricted=True, val_type=int, min_val=128, max_val=4058)
    check_argument('sample_rate', c['audio'], restricted=True, val_type=int, min_val=512, max_val=100000)
    check_argument('frame_length_ms', c['audio'], restricted=True, val_type=float, min_val=10, max_val=1000, alternative='win_length')
    check_argument('frame_shift_ms', c['audio'], restricted=True, val_type=float, min_val=1, max_val=1000, alternative='hop_length')
    check_argument('preemphasis', c['audio'], restricted=True, val_type=float, min_val=0, max_val=1)
    check_argument('min_level_db', c['audio'], restricted=True, val_type=int, min_val=-1000, max_val=10)
    check_argument('ref_level_db', c['audio'], restricted=True, val_type=int, min_val=0, max_val=1000)
    check_argument('power', c['audio'], restricted=True, val_type=float, min_val=1, max_val=5)
    check_argument('griffin_lim_iters', c['audio'], restricted=True, val_type=int, min_val=10, max_val=1000)

    # vocabulary parameters
    check_argument('characters', c, restricted=False, val_type=dict)
    check_argument('pad', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('eos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('bos', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('characters', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)
    check_argument('phonemes', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys() and c['use_phonemes'], val_type=str)
    check_argument('punctuations', c['characters'] if 'characters' in c.keys() else {}, restricted='characters' in c.keys(), val_type=str)

    # normalization parameters
    check_argument('signal_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('symmetric_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('max_norm', c['audio'], restricted=True, val_type=float, min_val=0.1, max_val=1000)
    check_argument('clip_norm', c['audio'], restricted=True, val_type=bool)
    check_argument('mel_fmin', c['audio'], restricted=True, val_type=float, min_val=0.0, max_val=1000)
    check_argument('mel_fmax', c['audio'], restricted=True, val_type=float, min_val=500.0)
    check_argument('spec_gain', c['audio'], restricted=True, val_type=[int, float], min_val=1, max_val=100)
    check_argument('do_trim_silence', c['audio'], restricted=True, val_type=bool)
    check_argument('trim_db', c['audio'], restricted=True, val_type=int)

    # training parameters
    check_argument('batch_size', c, restricted=True, val_type=int, min_val=1)
    check_argument('eval_batch_size', c, restricted=True, val_type=int, min_val=1)
    check_argument('r', c, restricted=True, val_type=int, min_val=1)
    check_argument('gradual_training', c, restricted=False, val_type=list)
    check_argument('mixed_precision', c, restricted=False, val_type=bool)
    # check_argument('grad_accum', c, restricted=True, val_type=int, min_val=1, max_val=100)

    # loss parameters
    check_argument('loss_masking', c, restricted=True, val_type=bool)
    if c['model'].lower() in ['tacotron', 'tacotron2']:
        check_argument('decoder_loss_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('postnet_loss_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('postnet_diff_spec_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('decoder_diff_spec_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('decoder_ssim_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('postnet_ssim_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('ga_alpha', c, restricted=True, val_type=float, min_val=0)
    if c['model'].lower in ["speedy_speech", "align_tts"]:
        check_argument('ssim_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('l1_alpha', c, restricted=True, val_type=float, min_val=0)
        check_argument('huber_alpha', c, restricted=True, val_type=float, min_val=0)

    # validation parameters
    check_argument('run_eval', c, restricted=True, val_type=bool)
    check_argument('test_delay_epochs', c, restricted=True, val_type=int, min_val=0)
    check_argument('test_sentences_file', c, restricted=False, val_type=str)

    # optimizer
    check_argument('noam_schedule', c, restricted=False, val_type=bool)
    check_argument('grad_clip', c, restricted=True, val_type=float, min_val=0.0)
    check_argument('epochs', c, restricted=True, val_type=int, min_val=1)
    check_argument('lr', c, restricted=True, val_type=float, min_val=0)
    check_argument('wd', c, restricted=is_tacotron(c), val_type=float, min_val=0)
    check_argument('warmup_steps', c, restricted=True, val_type=int, min_val=0)
    check_argument('seq_len_norm', c, restricted=is_tacotron(c), val_type=bool)

    # tacotron prenet
    check_argument('memory_size', c, restricted=is_tacotron(c), val_type=int, min_val=-1)
    check_argument('prenet_type', c, restricted=is_tacotron(c), val_type=str, enum_list=['original', 'bn'])
    check_argument('prenet_dropout', c, restricted=is_tacotron(c), val_type=bool)

    # attention
    check_argument('attention_type', c, restricted=is_tacotron(c), val_type=str, enum_list=['graves', 'original', 'dynamic_convolution'])
    check_argument('attention_heads', c, restricted=is_tacotron(c), val_type=int)
    check_argument('attention_norm', c, restricted=is_tacotron(c), val_type=str, enum_list=['sigmoid', 'softmax'])
    check_argument('windowing', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('use_forward_attn', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('forward_attn_mask', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('transition_agent', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('transition_agent', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('location_attn', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('bidirectional_decoder', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('double_decoder_consistency', c, restricted=is_tacotron(c), val_type=bool)
    check_argument('ddc_r', c, restricted='double_decoder_consistency' in c.keys(), min_val=1, max_val=7, val_type=int)

    if c['model'].lower() in ['tacotron', 'tacotron2']:
        # stopnet
        check_argument('stopnet', c, restricted=is_tacotron(c), val_type=bool)
        check_argument('separate_stopnet', c, restricted=is_tacotron(c), val_type=bool)

    # Model Parameters for non-tacotron models
    if c['model'].lower in ["speedy_speech", "align_tts"]:
        check_argument('positional_encoding', c, restricted=True, val_type=type)
        check_argument('encoder_type', c, restricted=True, val_type=str)
        check_argument('encoder_params', c, restricted=True, val_type=dict)
        check_argument('decoder_residual_conv_bn_params', c, restricted=True, val_type=dict)

    # GlowTTS parameters
    check_argument('encoder_type', c, restricted=not is_tacotron(c), val_type=str)

    # tensorboard
    check_argument('print_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('tb_plot_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('save_step', c, restricted=True, val_type=int, min_val=1)
    check_argument('checkpoint', c, restricted=True, val_type=bool)
    check_argument('tb_model_param_stats', c, restricted=True, val_type=bool)

    # dataloading
    # pylint: disable=import-outside-toplevel
    from TTS.tts.utils.text import cleaners
    check_argument('text_cleaner', c, restricted=True, val_type=str, enum_list=dir(cleaners))
    check_argument('enable_eos_bos_chars', c, restricted=True, val_type=bool)
    check_argument('num_loader_workers', c, restricted=True, val_type=int, min_val=0)
    check_argument('num_val_loader_workers', c, restricted=True, val_type=int, min_val=0)
    check_argument('batch_group_size', c, restricted=True, val_type=int, min_val=0)
    check_argument('min_seq_len', c, restricted=True, val_type=int, min_val=0)
    check_argument('max_seq_len', c, restricted=True, val_type=int, min_val=10)
    check_argument('compute_input_seq_cache', c, restricted=True, val_type=bool)

    # paths
    check_argument('output_path', c, restricted=True, val_type=str)

    # multi-speaker and gst
    check_argument('use_speaker_embedding', c, restricted=True, val_type=bool)
    check_argument('use_external_speaker_embedding_file', c, restricted=c['use_speaker_embedding'], val_type=bool)
    check_argument('external_speaker_embedding_file', c, restricted=c['use_external_speaker_embedding_file'], val_type=str)
    if c['model'].lower() in ['tacotron', 'tacotron2'] and c['use_gst']:
        check_argument('use_gst', c, restricted=is_tacotron(c), val_type=bool)
        check_argument('gst', c, restricted=is_tacotron(c), val_type=dict)
        check_argument('gst_style_input', c['gst'], restricted=is_tacotron(c), val_type=[str, dict])
        check_argument('gst_embedding_dim', c['gst'], restricted=is_tacotron(c), val_type=int, min_val=0, max_val=1000)
        check_argument('gst_use_speaker_embedding', c['gst'], restricted=is_tacotron(c), val_type=bool)
        check_argument('gst_num_heads', c['gst'], restricted=is_tacotron(c), val_type=int, min_val=2, max_val=10)
        check_argument('gst_style_tokens', c['gst'], restricted=is_tacotron(c), val_type=int, min_val=1, max_val=1000)

    # datasets - checking only the first entry
    check_argument('datasets', c, restricted=True, val_type=list)
    for dataset_entry in c['datasets']:
        check_argument('name', dataset_entry, restricted=True, val_type=str)
        check_argument('path', dataset_entry, restricted=True, val_type=str)
        check_argument('meta_file_train', dataset_entry, restricted=True, val_type=[str, list])
        check_argument('meta_file_val', dataset_entry, restricted=True, val_type=str)
