import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .common_layers import init_attn, Prenet, Linear


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(in_channels,
                           out_channels,
                           kernel_size,
                           padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        dropout = nn.Dropout(p=0.5)
        if nonlinear == 'relu':
            self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)

    def forward(self, x):
        output = self.net(x)
        return output


class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(mel_dim, 512, kernel_size=5, nonlinear='tanh'))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(
                ConvBNBlock(512, 512, kernel_size=5, nonlinear='tanh'))
        self.convolutions.append(
            ConvBNBlock(512, mel_dim, kernel_size=5, nonlinear=None))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_features=512):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(3):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu'))
        self.convolutions = nn.Sequential(*convolutions)
        self.lstm = nn.LSTM(in_features,
                            int(in_features / 2),
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.rnn_state = None

    def forward(self, x, input_lengths):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              input_lengths,
                                              batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs

    def inference(self, x):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs

    def inference_truncated(self, x):
        """
        Preserve encoder state for continuous inference
        """
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, self.rnn_state = self.lstm(x, self.rnn_state)
        return outputs


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, in_features, memory_dim, r, attn_type, attn_win, attn_norm,
                 prenet_type, prenet_dropout, forward_attn, trans_agent,
                 forward_attn_mask, location_attn, attn_K, separate_stopnet,
                 speaker_embedding_dim):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r_init = r
        self.r = r
        self.encoder_embedding_dim = in_features
        self.separate_stopnet = separate_stopnet
        self.query_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # memory -> |Prenet| -> processed_memory
        prenet_dim = self.memory_dim
        self.prenet = Prenet(
            prenet_dim,
            prenet_type,
            prenet_dropout,
            out_features=[self.prenet_dim, self.prenet_dim],
            bias=False)

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_features,
                                         self.query_dim)

        self.attention = init_attn(attn_type=attn_type,
                                   query_dim=self.query_dim,
                                   embedding_dim=in_features,
                                   attention_dim=128,
                                   location_attention=location_attn,
                                   attention_location_n_filters=32,
                                   attention_location_kernel_size=31,
                                   windowing=attn_win,
                                   norm=attn_norm,
                                   forward_attn=forward_attn,
                                   trans_agent=trans_agent,
                                   forward_attn_mask=forward_attn_mask,
                                   attn_K=attn_K)

        self.decoder_rnn = nn.LSTMCell(self.query_dim + in_features,
                                       self.decoder_rnn_dim, 1)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_features,
                                        self.memory_dim * self.r_init)

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(self.decoder_rnn_dim + self.memory_dim * self.r_init,
                   1,
                   bias=True,
                   init_gain='sigmoid'))
        self.memory_truncated = None

    def set_r(self, new_r):
        self.r = new_r

    def get_go_frame(self, inputs):
        B = inputs.size(0)
        memory = torch.zeros(1, device=inputs.device).repeat(B,
                             self.memory_dim * self.r)
        return memory

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)
        # T = inputs.size(1)
        if not keep_states:
            self.query = torch.zeros(1, device=inputs.device).repeat(
                B, self.query_dim)
            self.attention_rnn_cell_state = torch.zeros(
                1, device=inputs.device).repeat(B, self.query_dim)
            self.decoder_hidden = torch.zeros(1, device=inputs.device).repeat(
                B, self.decoder_rnn_dim)
            self.decoder_cell = torch.zeros(1, device=inputs.device).repeat(
                B, self.decoder_rnn_dim)
            self.context = torch.zeros(1, device=inputs.device).repeat(
                B, self.encoder_embedding_dim)
        self.inputs = inputs
        self.processed_inputs = self.attention.preprocess_inputs(inputs)
        self.mask = mask

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.memory_dim:
            memory = memory.view(memory.shape[0], memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, memory_dim)
        memory = memory.transpose(0, 1)
        return memory

    def _parse_outputs(self, outputs, stop_tokens, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.memory_dim)
        outputs = outputs.transpose(1, 2)
        return outputs, stop_tokens, alignments

    def _update_memory(self, memory):
        if len(memory.shape) == 2:
            return memory[:, self.memory_dim * (self.r - 1):]
        return memory[:, :, self.memory_dim * (self.r - 1):]

    def decode(self, memory):
        '''
         shapes:
            - memory: B x r * self.memory_dim
        '''
        # self.context: B x D_en
        # query_input: B x D_en + (r * self.memory_dim)
        query_input = torch.cat((memory, self.context), -1)
        # self.query and self.attention_rnn_cell_state : B x D_attn_rnn
        self.query, self.attention_rnn_cell_state = self.attention_rnn(
            query_input, (self.query, self.attention_rnn_cell_state))
        self.query = F.dropout(self.query, self.p_attention_dropout,
                               self.training)
        self.attention_rnn_cell_state = F.dropout(
            self.attention_rnn_cell_state, self.p_attention_dropout,
            self.training)
        # B x D_en
        self.context = self.attention(self.query, self.inputs,
                                      self.processed_inputs, self.mask)
        # B x (D_en + D_attn_rnn)
        decoder_rnn_input = torch.cat((self.query, self.context), -1)
        # self.decoder_hidden and self.decoder_cell: B x D_decoder_rnn
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        # B x (D_decoder_rnn + D_en)
        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context),
                                           dim=1)
        # B x (self.r * self.memory_dim)
        decoder_output = self.linear_projection(decoder_hidden_context)
        # B x (D_decoder_rnn + (self.r * self.memory_dim))
        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)
        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        # select outputs for the reduction rate self.r
        decoder_output = decoder_output[:, :self.r * self.memory_dim]
        return decoder_output, self.attention.attention_weights, stop_token

    def forward(self, inputs, memories, mask, speaker_embeddings=None):
        memory = self.get_go_frame(inputs).unsqueeze(0)
        memories = self._reshape_memory(memories)
        memories = torch.cat((memory, memories), dim=0)
        memories = self._update_memory(memories)
        if speaker_embeddings is not None:
            memories = torch.cat([memories, speaker_embeddings], dim=-1)
        memories = self.prenet(memories)

        self._init_states(inputs, mask=mask)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            decoder_output, attention_weights, stop_token = self.decode(memory)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [attention_weights]

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)
        return outputs, alignments, stop_tokens

    def inference(self, inputs, speaker_embeddings=None):
        memory = self.get_go_frame(inputs)
        memory = self._update_memory(memory)

        self._init_states(inputs, mask=None)
        self.attention.init_states(inputs)

        outputs, stop_tokens, alignments, t = [], [], [], 0
        while True:
            memory = self.prenet(memory)
            if speaker_embeddings is not None:
                memory = torch.cat([memory, speaker_embeddings], dim=-1)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > 0.7 and t > inputs.shape[0] / 2:
                break
            if len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

            memory = self._update_memory(decoder_output)
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_truncated(self, inputs):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None, keep_states=False)
        else:
            self._init_states(inputs, mask=None, keep_states=True)

        self.attention.init_win_idx()
        self.attention.init_states(inputs)
        outputs, stop_tokens, alignments, t = [], [], [], 0
        stop_flags = [True, False, False]
        while True:
            memory = self.prenet(self.memory_truncated)
            decoder_output, alignment, stop_token = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [decoder_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            if stop_token > 0.7:
                break
            if len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

            self.memory_truncated = decoder_output
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)

        return outputs, alignments, stop_tokens

    def inference_step(self, inputs, t, memory=None):
        """
        For debug purposes
        """
        if t == 0:
            memory = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None)

        memory = self.prenet(memory)
        decoder_output, stop_token, alignment = self.decode(memory)
        stop_token = torch.sigmoid(stop_token.data)
        memory = decoder_output
        return decoder_output, stop_token, alignment
