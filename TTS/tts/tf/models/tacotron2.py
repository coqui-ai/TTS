import tensorflow as tf
from tensorflow import keras

from TTS.tts.tf.layers.tacotron.tacotron2 import Encoder, Decoder, Postnet
from TTS.tts.tf.utils.tf_utils import shape_list


#pylint: disable=too-many-ancestors, abstract-method
class Tacotron2(keras.models.Model):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 postnet_output_dim=80,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 attn_norm="softmax",
                 attn_K=4,
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 enable_tflite=False):
        super(Tacotron2, self).__init__()
        self.r = r
        self.decoder_output_dim = decoder_output_dim
        self.postnet_output_dim = postnet_output_dim
        self.bidirectional_decoder = bidirectional_decoder
        self.num_speakers = num_speakers
        self.speaker_embed_dim = 256
        self.enable_tflite = enable_tflite

        self.embedding = keras.layers.Embedding(num_chars, 512, name='embedding')
        self.encoder = Encoder(512, name='encoder')
        # TODO: most of the decoder args have no use at the momment
        self.decoder = Decoder(decoder_output_dim,
                               r,
                               attn_type=attn_type,
                               use_attn_win=attn_win,
                               attn_norm=attn_norm,
                               prenet_type=prenet_type,
                               prenet_dropout=prenet_dropout,
                               use_forward_attn=forward_attn,
                               use_trans_agent=trans_agent,
                               use_forward_attn_mask=forward_attn_mask,
                               use_location_attn=location_attn,
                               attn_K=attn_K,
                               separate_stopnet=separate_stopnet,
                               speaker_emb_dim=self.speaker_embed_dim,
                               name='decoder',
                               enable_tflite=enable_tflite)
        self.postnet = Postnet(postnet_output_dim, 5, name='postnet')

    @tf.function(experimental_relax_shapes=True)
    def call(self, characters, text_lengths=None, frames=None, training=None):
        if training:
            return self.training(characters, text_lengths, frames)
        if not training:
            return self.inference(characters)
        raise RuntimeError(' [!] Set model training mode True or False')

    def training(self, characters, text_lengths, frames):
        B, T = shape_list(characters)
        embedding_vectors = self.embedding(characters, training=True)
        encoder_output = self.encoder(embedding_vectors, training=True)
        decoder_states = self.decoder.build_decoder_initial_states(B, 512, T)
        decoder_frames, stop_tokens, attentions = self.decoder(encoder_output, decoder_states, frames, text_lengths, training=True)
        postnet_frames = self.postnet(decoder_frames, training=True)
        output_frames = decoder_frames + postnet_frames
        return decoder_frames, output_frames, attentions, stop_tokens

    def inference(self, characters):
        B, T = shape_list(characters)
        embedding_vectors = self.embedding(characters, training=False)
        encoder_output = self.encoder(embedding_vectors, training=False)
        decoder_states = self.decoder.build_decoder_initial_states(B, 512, T)
        decoder_frames, stop_tokens, attentions = self.decoder(encoder_output, decoder_states, training=False)
        postnet_frames = self.postnet(decoder_frames, training=False)
        output_frames = decoder_frames + postnet_frames
        print(output_frames.shape)
        return decoder_frames, output_frames, attentions, stop_tokens

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([1, None], dtype=tf.int32),
        ],)
    def inference_tflite(self, characters):
        B, T = shape_list(characters)
        embedding_vectors = self.embedding(characters, training=False)
        encoder_output = self.encoder(embedding_vectors, training=False)
        decoder_states = self.decoder.build_decoder_initial_states(B, 512, T)
        decoder_frames, stop_tokens, attentions = self.decoder(encoder_output, decoder_states, training=False)
        postnet_frames = self.postnet(decoder_frames, training=False)
        output_frames = decoder_frames + postnet_frames
        print(output_frames.shape)
        return decoder_frames, output_frames, attentions, stop_tokens

    def build_inference(self, ):
        # TODO: issue https://github.com/PyCQA/pylint/issues/3613
        input_ids = tf.random.uniform(shape=[1, 4], maxval=10, dtype=tf.int32)  #pylint: disable=unexpected-keyword-arg
        self(input_ids)
