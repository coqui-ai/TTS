import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
# from tensorflow_addons.seq2seq import BahdanauAttention

# NOTE: linter has a problem with the current TF release
#pylint: disable=no-value-for-parameter
#pylint: disable=unexpected-keyword-arg

class Linear(keras.layers.Layer):
    def __init__(self, units, use_bias, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.linear_layer = keras.layers.Dense(units, use_bias=use_bias, name='linear_layer')
        self.activation = keras.layers.ReLU()

    def call(self, x):
        """
        shapes:
            x: B x T x C
        """
        return self.activation(self.linear_layer(x))


class LinearBN(keras.layers.Layer):
    def __init__(self, units, use_bias, **kwargs):
        super(LinearBN, self).__init__(**kwargs)
        self.linear_layer = keras.layers.Dense(units, use_bias=use_bias, name='linear_layer')
        self.batch_normalization = keras.layers.BatchNormalization(axis=-1, momentum=0.90, epsilon=1e-5, name='batch_normalization')
        self.activation = keras.layers.ReLU()

    def call(self, x, training=None):
        """
        shapes:
            x: B x T x C
        """
        out = self.linear_layer(x)
        out = self.batch_normalization(out, training=training)
        return self.activation(out)


class Prenet(keras.layers.Layer):
    def __init__(self,
                 prenet_type,
                 prenet_dropout,
                 units,
                 bias,
                 **kwargs):
        super(Prenet, self).__init__(**kwargs)
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        self.linear_layers = []
        if prenet_type == "bn":
            self.linear_layers += [LinearBN(unit, use_bias=bias, name=f'linear_layer_{idx}') for idx, unit in enumerate(units)]
        elif prenet_type == "original":
            self.linear_layers += [Linear(unit, use_bias=bias, name=f'linear_layer_{idx}') for idx, unit in enumerate(units)]
        else:
            raise RuntimeError(' [!] Unknown prenet type.')
        if prenet_dropout:
            self.dropout = keras.layers.Dropout(rate=0.5)

    def call(self, x, training=None):
        """
        shapes:
            x: B x T x C
        """
        for linear in self.linear_layers:
            if self.prenet_dropout:
                x = self.dropout(linear(x), training=training)
            else:
                x = linear(x)
        return x


def _sigmoid_norm(score):
    attn_weights = tf.nn.sigmoid(score)
    attn_weights = attn_weights / tf.reduce_sum(attn_weights, axis=1, keepdims=True)
    return attn_weights


class Attention(keras.layers.Layer):
    """TODO: implement forward_attention
    TODO: location sensitive attention
    TODO: implement attention windowing """
    def __init__(self, attn_dim, use_loc_attn, loc_attn_n_filters,
                 loc_attn_kernel_size, use_windowing, norm, use_forward_attn,
                 use_trans_agent, use_forward_attn_mask, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.use_loc_attn = use_loc_attn
        self.loc_attn_n_filters = loc_attn_n_filters
        self.loc_attn_kernel_size = loc_attn_kernel_size
        self.use_windowing = use_windowing
        self.norm = norm
        self.use_forward_attn = use_forward_attn
        self.use_trans_agent = use_trans_agent
        self.use_forward_attn_mask = use_forward_attn_mask
        self.query_layer = tf.keras.layers.Dense(attn_dim, use_bias=False, name='query_layer/linear_layer')
        self.inputs_layer = tf.keras.layers.Dense(attn_dim, use_bias=False, name=f'{self.name}/inputs_layer/linear_layer')
        self.v = tf.keras.layers.Dense(1, use_bias=True, name='v/linear_layer')
        if use_loc_attn:
            self.location_conv1d = keras.layers.Conv1D(
                filters=loc_attn_n_filters,
                kernel_size=loc_attn_kernel_size,
                padding='same',
                use_bias=False,
                name='location_layer/location_conv1d')
            self.location_dense = keras.layers.Dense(attn_dim, use_bias=False, name='location_layer/location_dense')
        if norm == 'softmax':
            self.norm_func = tf.nn.softmax
        elif norm == 'sigmoid':
            self.norm_func = _sigmoid_norm
        else:
            raise ValueError("Unknown value for attention norm type")

    def init_states(self, batch_size, value_length):
        states = []
        if self.use_loc_attn:
            attention_cum = tf.zeros([batch_size, value_length])
            attention_old = tf.zeros([batch_size, value_length])
            states = [attention_cum, attention_old]
        if self.use_forward_attn:
            alpha = tf.concat([
                tf.ones([batch_size, 1]),
                tf.zeros([batch_size, value_length])[:, :-1] + 1e-7
            ], 1)
            states.append(alpha)
        return tuple(states)

    def process_values(self, values):
        """ cache values for decoder iterations """
        #pylint: disable=attribute-defined-outside-init
        self.processed_values = self.inputs_layer(values)
        self.values = values

    def get_loc_attn(self, query, states):
        """ compute location attention, query layer and
        unnorm. attention weights"""
        attention_cum, attention_old = states[:2]
        attn_cat = tf.stack([attention_old, attention_cum], axis=2)

        processed_query = self.query_layer(tf.expand_dims(query, 1))
        processed_attn = self.location_dense(self.location_conv1d(attn_cat))
        score = self.v(
            tf.nn.tanh(self.processed_values + processed_query +
                       processed_attn))
        score = tf.squeeze(score, axis=2)
        return score, processed_query

    def get_attn(self, query):
        """ compute query layer and unnormalized attention weights """
        processed_query = self.query_layer(tf.expand_dims(query, 1))
        score = self.v(tf.nn.tanh(self.processed_values + processed_query))
        score = tf.squeeze(score, axis=2)
        return score, processed_query

    def apply_score_masking(self, score, mask):  #pylint: disable=no-self-use
        """ ignore sequence paddings """
        padding_mask = tf.expand_dims(math_ops.logical_not(mask), 2)
        # Bias so padding positions do not contribute to attention distribution.
        score -= 1.e9 * math_ops.cast(padding_mask, dtype=tf.float32)
        return score

    def apply_forward_attention(self, alignment, alpha):  #pylint: disable=no-self-use
        # forward attention
        fwd_shifted_alpha = tf.pad(alpha[:, :-1], ((0, 0), (1, 0)), constant_values=0.0)
        # compute transition potentials
        new_alpha = ((1 - 0.5) * alpha + 0.5 * fwd_shifted_alpha + 1e-8) * alignment
        # renormalize attention weights
        new_alpha = new_alpha / tf.reduce_sum(new_alpha, axis=1, keepdims=True)
        return new_alpha

    def update_states(self, old_states, scores_norm, attn_weights, new_alpha=None):
        states = []
        if self.use_loc_attn:
            states = [old_states[0] + scores_norm, attn_weights]
        if self.use_forward_attn:
            states.append(new_alpha)
        return tuple(states)

    def call(self, query, states):
        """
        shapes:
            query: B x D
        """
        if self.use_loc_attn:
            score, _ = self.get_loc_attn(query, states)
        else:
            score, _ = self.get_attn(query)

        # TODO: masking
        # if mask is not None:
        # self.apply_score_masking(score, mask)
        # attn_weights shape == (batch_size, max_length, 1)

        # normalize attention scores
        scores_norm = self.norm_func(score)
        attn_weights = scores_norm

        # apply forward attention
        new_alpha = None
        if self.use_forward_attn:
            new_alpha = self.apply_forward_attention(attn_weights, states[-1])
            attn_weights = new_alpha

        # update states tuple
        # states = (cum_attn_weights, attn_weights, new_alpha)
        states = self.update_states(states, scores_norm, attn_weights, new_alpha)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.matmul(tf.expand_dims(attn_weights, axis=2), self.values, transpose_a=True, transpose_b=False)
        context_vector = tf.squeeze(context_vector, axis=1)
        return context_vector, attn_weights, states


# def _location_sensitive_score(processed_query, keys, processed_loc, attention_v, attention_b):
#     dtype = processed_query.dtype
#     num_units = keys.shape[-1].value or array_ops.shape(keys)[-1]
#     return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query + processed_loc + attention_b), [2])


# class LocationSensitiveAttention(BahdanauAttention):
#     def __init__(self,
#                  units,
#                  memory=None,
#                  memory_sequence_length=None,
#                  normalize=False,
#                  probability_fn="softmax",
#                  kernel_initializer="glorot_uniform",
#                  dtype=None,
#                  name="LocationSensitiveAttention",
#                  location_attention_filters=32,
#                  location_attention_kernel_size=31):

#         super(LocationSensitiveAttention,
#                     self).__init__(units=units,
#                                     memory=memory,
#                                     memory_sequence_length=memory_sequence_length,
#                                     normalize=normalize,
#                                     probability_fn='softmax',  ## parent module default
#                                     kernel_initializer=kernel_initializer,
#                                     dtype=dtype,
#                                     name=name)
#         if probability_fn == 'sigmoid':
#             self.probability_fn = lambda score, _: self._sigmoid_normalization(score)
#         self.location_conv = keras.layers.Conv1D(filters=location_attention_filters, kernel_size=location_attention_kernel_size, padding='same', use_bias=False)
#         self.location_dense = keras.layers.Dense(units, use_bias=False)
#         # self.v = keras.layers.Dense(1, use_bias=True)

#     def  _location_sensitive_score(self, processed_query, keys, processed_loc):
#         processed_query = tf.expand_dims(processed_query, 1)
#         return tf.reduce_sum(self.attention_v * tf.tanh(keys + processed_query + processed_loc), [2])

#     def _location_sensitive(self, alignment_cum, alignment_old):
#         alignment_cat = tf.stack([alignment_cum, alignment_old], axis=2)
#         return self.location_dense(self.location_conv(alignment_cat))

#     def _sigmoid_normalization(self, score):
#         return tf.nn.sigmoid(score) / tf.reduce_sum(tf.nn.sigmoid(score), axis=-1, keepdims=True)

#     # def _apply_masking(self, score, mask):
#     #     padding_mask = tf.expand_dims(math_ops.logical_not(mask), 2)
#     #     # Bias so padding positions do not contribute to attention distribution.
#     #     score -= 1.e9 * math_ops.cast(padding_mask, dtype=tf.float32)
#     #     return score

#     def _calculate_attention(self, query, state):
#         alignment_cum, alignment_old = state[:2]
#         processed_query = self.query_layer(
#             query) if self.query_layer else query
#         processed_loc = self._location_sensitive(alignment_cum, alignment_old)
#         score = self._location_sensitive_score(
#             processed_query,
#             self.keys,
#             processed_loc)
#         alignment = self.probability_fn(score, state)
#         alignment_cum = alignment_cum + alignment
#         state[0] = alignment_cum
#         state[1] = alignment
#         return alignment, state

#     def compute_context(self, alignments):
#         expanded_alignments = tf.expand_dims(alignments, 1)
#         context = tf.matmul(expanded_alignments, self.values)
#         context = tf.squeeze(context, [1])
#         return context

#     # def call(self, query, state):
#     #     alignment, next_state = self._calculate_attention(query, state)
#     #     return alignment, next_state
