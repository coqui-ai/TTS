import tensorflow as tf
from tensorflow import keras
from TTS.tts.tf.utils.tf_utils import shape_list
from TTS.tts.tf.layers.tacotron.common_layers import Prenet, Attention


# NOTE: linter has a problem with the current TF release
#pylint: disable=no-value-for-parameter
#pylint: disable=unexpected-keyword-arg
class ConvBNBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, **kwargs):
        super(ConvBNBlock, self).__init__(**kwargs)
        self.convolution1d = keras.layers.Conv1D(filters, kernel_size, padding='same', name='convolution1d')
        self.batch_normalization = keras.layers.BatchNormalization(axis=2, momentum=0.90, epsilon=1e-5, name='batch_normalization')
        self.dropout = keras.layers.Dropout(rate=0.5, name='dropout')
        self.activation = keras.layers.Activation(activation, name='activation')

    def call(self, x, training=None):
        o = self.convolution1d(x)
        o = self.batch_normalization(o, training=training)
        o = self.activation(o)
        o = self.dropout(o, training=training)
        return o


class Postnet(keras.layers.Layer):
    def __init__(self, output_filters, num_convs, **kwargs):
        super(Postnet, self).__init__(**kwargs)
        self.convolutions = []
        self.convolutions.append(ConvBNBlock(512, 5, 'tanh', name='convolutions_0'))
        for idx in range(1, num_convs - 1):
            self.convolutions.append(ConvBNBlock(512, 5, 'tanh', name=f'convolutions_{idx}'))
        self.convolutions.append(ConvBNBlock(output_filters, 5, 'linear', name=f'convolutions_{idx+1}'))

    def call(self, x, training=None):
        o = x
        for layer in self.convolutions:
            o = layer(o, training=training)
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, output_input_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.convolutions = []
        for idx in range(3):
            self.convolutions.append(ConvBNBlock(output_input_dim, 5, 'relu', name=f'convolutions_{idx}'))
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(output_input_dim // 2, return_sequences=True, use_bias=True), name='lstm')

    def call(self, x, training=None):
        o = x
        for layer in self.convolutions:
            o = layer(o, training=training)
        o = self.lstm(o)
        return o


class Decoder(keras.layers.Layer):
    #pylint: disable=unused-argument
    def __init__(self, frame_dim, r, attn_type, use_attn_win, attn_norm, prenet_type,
                 prenet_dropout, use_forward_attn, use_trans_agent, use_forward_attn_mask,
                 use_location_attn, attn_K, separate_stopnet, speaker_emb_dim, enable_tflite, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.frame_dim = frame_dim
        self.r_init = tf.constant(r, dtype=tf.int32)
        self.r = tf.constant(r, dtype=tf.int32)
        self.output_dim = r * self.frame_dim
        self.separate_stopnet = separate_stopnet
        self.enable_tflite = enable_tflite

        # layer constants
        self.max_decoder_steps = tf.constant(1000, dtype=tf.int32)
        self.stop_thresh = tf.constant(0.5, dtype=tf.float32)

        # model dimensions
        self.query_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.attn_dim = 128
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim],
                             bias=False,
                             name='prenet')
        self.attention_rnn = keras.layers.LSTMCell(self.query_dim, use_bias=True, name='attention_rnn', )
        self.attention_rnn_dropout = keras.layers.Dropout(0.5)

        # TODO: implement other attn options
        self.attention = Attention(attn_dim=self.attn_dim,
                                   use_loc_attn=True,
                                   loc_attn_n_filters=32,
                                   loc_attn_kernel_size=31,
                                   use_windowing=False,
                                   norm=attn_norm,
                                   use_forward_attn=use_forward_attn,
                                   use_trans_agent=use_trans_agent,
                                   use_forward_attn_mask=use_forward_attn_mask,
                                   name='attention')
        self.decoder_rnn = keras.layers.LSTMCell(self.decoder_rnn_dim, use_bias=True, name='decoder_rnn')
        self.decoder_rnn_dropout = keras.layers.Dropout(0.5)
        self.linear_projection = keras.layers.Dense(self.frame_dim * r, name='linear_projection/linear_layer')
        self.stopnet = keras.layers.Dense(1, name='stopnet/linear_layer')


    def set_max_decoder_steps(self, new_max_steps):
        self.max_decoder_steps = tf.constant(new_max_steps, dtype=tf.int32)

    def set_r(self, new_r):
        self.r = tf.constant(new_r, dtype=tf.int32)
        self.output_dim = self.frame_dim * new_r

    def build_decoder_initial_states(self, batch_size, memory_dim, memory_length):
        zero_frame = tf.zeros([batch_size, self.frame_dim])
        zero_context = tf.zeros([batch_size, memory_dim])
        attention_rnn_state = self.attention_rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        decoder_rnn_state = self.decoder_rnn.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        attention_states = self.attention.init_states(batch_size, memory_length)
        return zero_frame, zero_context, attention_rnn_state, decoder_rnn_state, attention_states

    def step(self, prenet_next, states,
             memory_seq_length=None, training=None):
        _, context_next, attention_rnn_state, decoder_rnn_state, attention_states = states
        attention_rnn_input = tf.concat([prenet_next, context_next], -1)
        attention_rnn_output, attention_rnn_state = \
                self.attention_rnn(attention_rnn_input,
                                   attention_rnn_state, training=training)
        attention_rnn_output = self.attention_rnn_dropout(attention_rnn_output, training=training)
        context, attention, attention_states = self.attention(attention_rnn_output, attention_states, training=training)
        decoder_rnn_input = tf.concat([attention_rnn_output, context], -1)
        decoder_rnn_output, decoder_rnn_state = \
                self.decoder_rnn(decoder_rnn_input, decoder_rnn_state, training=training)
        decoder_rnn_output = self.decoder_rnn_dropout(decoder_rnn_output, training=training)
        linear_projection_input = tf.concat([decoder_rnn_output, context], -1)
        output_frame = self.linear_projection(linear_projection_input, training=training)
        stopnet_input = tf.concat([decoder_rnn_output, output_frame], -1)
        stopnet_output = self.stopnet(stopnet_input, training=training)
        output_frame = output_frame[:, :self.r * self.frame_dim]
        states = (output_frame[:, self.frame_dim * (self.r - 1):], context, attention_rnn_state, decoder_rnn_state, attention_states)
        return output_frame, stopnet_output, states, attention

    def decode(self, memory, states, frames, memory_seq_length=None):
        B, _, _ = shape_list(memory)
        num_iter = shape_list(frames)[1] // self.r
        # init states
        frame_zero = tf.expand_dims(states[0], 1)
        frames = tf.concat([frame_zero, frames], axis=1)
        outputs = tf.TensorArray(dtype=tf.float32, size=num_iter)
        attentions = tf.TensorArray(dtype=tf.float32, size=num_iter)
        stop_tokens = tf.TensorArray(dtype=tf.float32, size=num_iter)
        # pre-computes
        self.attention.process_values(memory)
        prenet_output = self.prenet(frames, training=True)
        step_count = tf.constant(0, dtype=tf.int32)

        def _body(step, memory, prenet_output, states, outputs, stop_tokens, attentions):
            prenet_next = prenet_output[:, step]
            output, stop_token, states, attention = self.step(prenet_next,
                                                              states,
                                                              memory_seq_length)
            outputs = outputs.write(step, output)
            attentions = attentions.write(step, attention)
            stop_tokens = stop_tokens.write(step, stop_token)
            return step + 1, memory, prenet_output, states, outputs, stop_tokens, attentions
        _, memory, _, states, outputs, stop_tokens, attentions = \
                tf.while_loop(lambda *arg: True,
                              _body,
                              loop_vars=(step_count, memory, prenet_output,
                                         states, outputs, stop_tokens, attentions),
                              parallel_iterations=32,
                              swap_memory=True,
                              maximum_iterations=num_iter)

        outputs = outputs.stack()
        attentions = attentions.stack()
        stop_tokens = stop_tokens.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        attentions = tf.transpose(attentions, [1, 0, 2])
        stop_tokens = tf.transpose(stop_tokens, [1, 0, 2])
        stop_tokens = tf.squeeze(stop_tokens, axis=2)
        outputs = tf.reshape(outputs, [B, -1, self.frame_dim])
        return outputs, stop_tokens, attentions

    def decode_inference(self, memory, states):
        B, _, _ = shape_list(memory)
        # init states
        outputs = tf.TensorArray(dtype=tf.float32, size=0, clear_after_read=False, dynamic_size=True)
        attentions = tf.TensorArray(dtype=tf.float32, size=0, clear_after_read=False, dynamic_size=True)
        stop_tokens = tf.TensorArray(dtype=tf.float32, size=0, clear_after_read=False, dynamic_size=True)

        # pre-computes
        self.attention.process_values(memory)

        # iter vars
        stop_flag = tf.constant(False, dtype=tf.bool)
        step_count = tf.constant(0, dtype=tf.int32)

        def _body(step, memory, states, outputs, stop_tokens, attentions, stop_flag):
            frame_next = states[0]
            prenet_next = self.prenet(frame_next, training=False)
            output, stop_token, states, attention = self.step(prenet_next,
                                                              states,
                                                              None,
                                                              training=False)
            stop_token = tf.math.sigmoid(stop_token)
            outputs = outputs.write(step, output)
            attentions = attentions.write(step, attention)
            stop_tokens = stop_tokens.write(step, stop_token)
            stop_flag = tf.greater(stop_token, self.stop_thresh)
            stop_flag = tf.reduce_all(stop_flag)
            return step + 1, memory, states, outputs, stop_tokens, attentions, stop_flag

        cond = lambda step, m, s, o, st, a, stop_flag: tf.equal(stop_flag, tf.constant(False, dtype=tf.bool))
        _, memory, states, outputs, stop_tokens, attentions, stop_flag = \
                tf.while_loop(cond,
                              _body,
                              loop_vars=(step_count, memory, states, outputs,
                                         stop_tokens, attentions, stop_flag),
                              parallel_iterations=32,
                              swap_memory=True,
                              maximum_iterations=self.max_decoder_steps)

        outputs = outputs.stack()
        attentions = attentions.stack()
        stop_tokens = stop_tokens.stack()

        outputs = tf.transpose(outputs, [1, 0, 2])
        attentions = tf.transpose(attentions, [1, 0, 2])
        stop_tokens = tf.transpose(stop_tokens, [1, 0, 2])
        stop_tokens = tf.squeeze(stop_tokens, axis=2)
        outputs = tf.reshape(outputs, [B, -1, self.frame_dim])
        return outputs, stop_tokens, attentions

    def decode_inference_tflite(self, memory, states):
        """Inference with TF-Lite compatibility. It assumes
        batch_size is 1"""
        # init states
        # dynamic_shape is not supported in TFLite
        outputs = tf.TensorArray(dtype=tf.float32,
                                 size=self.max_decoder_steps,
                                 element_shape=tf.TensorShape(
                                     [self.output_dim]),
                                 clear_after_read=False,
                                 dynamic_size=False)
        # stop_flags = tf.TensorArray(dtype=tf.bool,
        #                          size=self.max_decoder_steps,
        #                          element_shape=tf.TensorShape(
        #                              []),
        #                          clear_after_read=False,
        #                          dynamic_size=False)
        attentions = ()
        stop_tokens = ()

        # pre-computes
        self.attention.process_values(memory)

        # iter vars
        stop_flag = tf.constant(False, dtype=tf.bool)
        step_count = tf.constant(0, dtype=tf.int32)

        def _body(step, memory, states, outputs, stop_flag):
            frame_next = states[0]
            prenet_next = self.prenet(frame_next, training=False)
            output, stop_token, states, _ = self.step(prenet_next,
                                                      states,
                                                      None,
                                                      training=False)
            stop_token = tf.math.sigmoid(stop_token)
            stop_flag = tf.greater(stop_token, self.stop_thresh)
            stop_flag = tf.reduce_all(stop_flag)
            # stop_flags = stop_flags.write(step, tf.logical_not(stop_flag))

            outputs = outputs.write(step, tf.reshape(output, [-1]))
            return step + 1, memory, states, outputs, stop_flag

        cond = lambda step, m, s, o, stop_flag: tf.equal(stop_flag, tf.constant(False, dtype=tf.bool))
        step_count, memory, states, outputs, stop_flag = \
                tf.while_loop(cond,
                              _body,
                              loop_vars=(step_count, memory, states, outputs,
                                         stop_flag),
                              parallel_iterations=32,
                              swap_memory=True,
                              maximum_iterations=self.max_decoder_steps)


        outputs = outputs.stack()
        outputs = tf.gather(outputs, tf.range(step_count)) # pylint: disable=no-value-for-parameter
        outputs = tf.expand_dims(outputs, axis=[0])
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [1, -1, self.frame_dim])
        return outputs, stop_tokens, attentions


    def call(self, memory, states, frames=None, memory_seq_length=None, training=False):
        if training:
            return self.decode(memory, states, frames, memory_seq_length)
        if self.enable_tflite:
            return self.decode_inference_tflite(memory, states)
        return self.decode_inference(memory, states)
