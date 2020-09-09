import datetime
import importlib
import pickle
import numpy as np
import tensorflow as tf


def save_checkpoint(model, optimizer, current_step, epoch, r, output_path, **kwargs):
    state = {
        'model': model.weights,
        'optimizer': optimizer,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': r
    }
    state.update(kwargs)
    pickle.dump(state, open(output_path, 'wb'))


def load_checkpoint(model, checkpoint_path):
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    chkp_var_dict = {var.name: var.numpy() for var in checkpoint['model']}
    tf_vars = model.weights
    for tf_var in tf_vars:
        layer_name = tf_var.name
        try:
            chkp_var_value = chkp_var_dict[layer_name]
        except KeyError:
            class_name = list(chkp_var_dict.keys())[0].split("/")[0]
            layer_name = f"{class_name}/{layer_name}"
            chkp_var_value = chkp_var_dict[layer_name]

        tf.keras.backend.set_value(tf_var, chkp_var_value)
    if 'r' in checkpoint.keys():
        model.decoder.set_r(checkpoint['r'])
    return model


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = np.empty([0, max_len], dtype=np.int8)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand


# @tf.custom_gradient
def check_gradient(x, grad_clip):
    x_normed = tf.clip_by_norm(x, grad_clip)
    grad_norm = tf.norm(grad_clip)
    return x_normed, grad_norm


def count_parameters(model, c):
    try:
        return model.count_params()
    except RuntimeError:
        input_dummy = tf.convert_to_tensor(np.random.rand(8, 128).astype('int32'))
        input_lengths = np.random.randint(100, 129, (8, ))
        input_lengths[-1] = 128
        input_lengths = tf.convert_to_tensor(input_lengths.astype('int32'))
        mel_spec = np.random.rand(8, 2 * c.r,
                                  c.audio['num_mels']).astype('float32')
        mel_spec = tf.convert_to_tensor(mel_spec)
        speaker_ids = np.random.randint(
            0, 5, (8, )) if c.use_speaker_embedding else None
        _ = model(input_dummy, input_lengths, mel_spec, speaker_ids=speaker_ids)
        return model.count_params()


def setup_model(num_chars, num_speakers, c, enable_tflite=False):
    print(" > Using model: {}".format(c.model))
    MyModel = importlib.import_module('TTS.tts.tf.models.' + c.model.lower())
    MyModel = getattr(MyModel, c.model)
    if c.model.lower() in "tacotron":
        raise NotImplementedError(' [!] Tacotron model is not ready.')
    # tacotron2
    model = MyModel(num_chars=num_chars,
                    num_speakers=num_speakers,
                    r=c.r,
                    postnet_output_dim=c.audio['num_mels'],
                    decoder_output_dim=c.audio['num_mels'],
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
                    enable_tflite=enable_tflite)
    return model
