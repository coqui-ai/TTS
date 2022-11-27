from dataclasses import dataclass

import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

from TTS.tts.layers.tacotron.common_layers import Prenet
from TTS.tts.utils.helpers import log_clamped, logsumexp, sequence_mask


class HMM(nn.Module):
    def __init__(self,
                 frame_channels,
                 ar_order,
                 prenet_type,
                 prenet_dim,
                 prenet_dropout,
                 memory_rnn_dim,
                 prenet_dropout_while_eval,
                 ar_data_dropout=False,
    ):
        super().__init__()
        
        self.transition_model = TransitionModel()
        self.emission_model = EmissionModel()
        
        assert ar_order > 0, f"AR order must be greater than 0 provided {ar_order}"
        
        self.ar_order = ar_order
        self.prenet = Prenet(
            in_features=frame_channels * ar_order,
            prenet_type=prenet_type,
            prenet_dropout=prenet_dropout,
            out_features=[self.prenet_dim, self.prenet_dim],
            bias=False
        )
        self.memory_rnn = nn.LSTMCell(input_size=prenet_dim, hidden_size=memory_rnn_dim)
        self.output_net = OutputNet(hparams)
        self.register_buffer("go_tokens", torch.zeros(ar_order, 1))

    def forward(self, inputs, inputs_len, mels, mel_lens):
        r"""
        HMM forward algorithm for training

        Args:
            text_embs (torch.FloatTensor): Encoder outputs
            text_lens (torch.LongTensor): Encoder output lengths
            mels (torch.FloatTensor): Mel inputs for teacher forcing
            mel_lens (torch.LongTensor): Length of mel inputs
        
        Shapes:
            - text_embs: (batch, C, T)
            - 

        Returns:
            log_prob (torch.FloatTensor): Log probability of the sequence
        """

        # Get dimensions of inputs
        batch_size, self.N = mels.shape[0]
        T_max = torch.max(mel_lens)
        self.N = inputs.shape[1]
        mels = mels.permute(0, 2, 1) #! TODO: check dataloader here

        # Intialize forward algorithm
        log_state_priors = self._initialize_log_state_priors(inputs)
        log_c = self._initialize_forward_algorithm_variables(mels)

        # Initialize autoregression elements
        ar_inputs = self._add_go_token(mels)
        h_post_prenet, c_post_prenet = self.init_lstm_states(batch_size, self.hparams.post_prenet_rnn_dim, mels)

        for t in range(T_max):

            # Process Autoregression
            h_post_prenet, c_post_prenet = self.process_ar_timestep(
                t,
                ar_inputs,
                h_post_prenet,
                c_post_prenet,
                data_dropout_flag,
                prenet_dropout_flag,
            )

            # Get mean, std and transition vector from decoder for this timestep
            mean, std, transition_vector = self.decoder(h_post_prenet, inputs)

            # Forward algorithm for this timestep
            if t == 0:
                log_alpha_temp = log_state_priors + self.emission_model(mels[:, 0], mean, std, inputs_len)
            else:
                log_alpha_temp = self.emission_model(mels[:, t], mean, std, inputs_len) + self.transition_model(
                    self.log_alpha_scaled[:, t - 1, :], transition_vector, inputs_len
                )

            log_c[:, t] = torch.logsumexp(log_alpha_temp, dim=1)
            self.log_alpha_scaled[:, t, :] = log_alpha_temp - log_c[:, t].unsqueeze(1)

            # Save for plotting
            self.transition_vector[:, t] = transition_vector.detach()
            self.means.append(mean.detach())

        log_c = self.mask_lengths(mels, mel_lens, log_c)

        sum_final_log_c = self.get_absorption_state_scaling_factor(
            mel_lens, self.log_alpha_scaled, inputs_len
        )

        log_probs = torch.sum(log_c, dim=1) + sum_final_log_c

        return log_probs

    def mask_lengths(self, mel_inputs, mel_inputs_lengths, log_c):
        """
        Mask the lengths of the forward variables so that the variable lenghts
        do not contribute in the loss calculation
        Args:
            mel_inputs (torch.FloatTensor): (batch, T, frame_channels)
            mel_inputs_lengths (torch.IntTensor): (batch)
            log_c (torch.FloatTensor): (batch, T)
        Returns:
            log_c (torch.FloatTensor) : scaled probabilities (batch, T)
        """
        batch_size, T, n_mel_channel = mel_inputs.shape
        # create len mask
        mask_tensor = mel_inputs.new_zeros(T)
        mask_log_c = torch.arange(T, out=mask_tensor).expand(len(mel_inputs_lengths), T) < (
            mel_inputs_lengths
        ).unsqueeze(1)
        # mask log_c
        log_c = log_c * mask_log_c

        # mask log_alpha_scaled
        mask_log_alpha_scaled = mask_log_c.unsqueeze(2)
        self.log_alpha_scaled = self.log_alpha_scaled * mask_log_alpha_scaled

        return log_c

    def process_ar_timestep(
        self,
        t,
        ar_inputs,
        h_post_prenet,
        c_post_prenet,
        data_dropout_flag,
        prenet_dropout_flag,
    ):
        """
        Process autoregression in timestep
        1. At a specific t timestep
        2. Perform data dropout if applied (we did not use it)
        3. Run the autoregressive frame through the prenet (has dropout)
        4. Run the prenet output through the post prenet rnn

        Args:
            t (int): mel-spec timestep
            ar_inputs (torch.FloatTensor): go-token appended mel-spectrograms
            h_post_prenet (torch.FloatTensor): previous timestep rnn hidden state
            c_post_prenet (torch.FloatTensor): previous timestep rnn cell state
            data_dropout_flag (bool): data dropout flag
            prenet_dropout_flag (bool): data dropout flag

        Returns:
            h_post_prenet (torch.FloatTensor): rnn hidden state of the current timestep
            c_post_prenet (torch.FloatTensor): rnn cell state of the current timestep
        """
        prenet_input = self.perform_data_dropout_of_ar_mel_inputs(
            ar_inputs[:, t : t + self.hparams.ar_order], data_dropout_flag
        )
        ar_inputs_prenet = self.prenet(prenet_input.flatten(1), prenet_dropout_flag)
        h_post_prenet, c_post_prenet = self.post_prenet_rnn(ar_inputs_prenet, (h_post_prenet, c_post_prenet))
        return h_post_prenet, c_post_prenet

    def _add_go_token(self, mel_inputs):
        """Append the go token to create the autoregressive input
        Args:
            mel_inputs (torch.FloatTensor): (batch_size, T, n_mel_channel)
        Returns:
            ar_inputs (torch.FloatTensor): (batch_size, T, n_mel_channel)
        """
        batch_size, T, _ = mel_inputs.shape
        go_tokens = self.go_tokens.unsqueeze(0).expand(
            batch_size, self.ar_order, self.frame_channels
        )
        ar_inputs = torch.cat((go_tokens, mel_inputs), dim=1)[:, :T]
        return ar_inputs

    def _initialize_forward_algorithm_variables(self, mel_inputs):
        r"""Initialize placeholders for forward algorithm variables, to use a stable
                version we will use log_alpha_scaled and the scaling constant

        Args:
            mel_inputs (torch.FloatTensor): (b, T, frame_channels)
        Returns:
            log_c (torch.FloatTensor): Scaling constant (b, T)
        """
        batch_size, T_max, _ = mel_inputs.shape
        self.log_alpha_scaled = mel_inputs.new_zeros((batch_size, T_max, self.N))
        log_c = mel_inputs.new_zeros(batch_size, T_max)

        # Saving for plotting later, will not have gradient tapes
        self.means = []
        self.transition_vector = mel_inputs.new_zeros((batch_size, T_max, self.N))
        return log_c

    def init_lstm_states(self, batch_size, hidden_state_dim, device_tensor):
        r"""
        Initialize Hidden and Cell states for LSTM Cell

        Args:
            batch_size (Int): batch size
            hidden_state_dim (Int): dimensions of the h and c
            device_tensor (torch.FloatTensor): useful for the device and type

        Returns:
            (torch.FloatTensor): shape (batch_size, hidden_state_dim)
                can be hidden state for LSTM
            (torch.FloatTensor): shape (batch_size, hidden_state_dim)
                can be the cell state for LSTM
        """
        return (
            device_tensor.new_zeros(batch_size, hidden_state_dim),
            device_tensor.new_zeros(batch_size, hidden_state_dim),
        )

    def perform_data_dropout_of_ar_mel_inputs(self, mel_inputs, dropout_flag):
        r"""
        Takes mel frames as inputs and applies data dropout on it

        Args:
            mel_inputs (torch.FloatTensor): mel frames
                shape: (batch, ar_order, frame_channels)
            dropout_flag (Bool): to dropout during eval or not

        Returns:
            mel_inputs (torch.FloatTensor) : data droped out mel frames
        """
        batch_size, n_frame_per_step, _ = mel_inputs.shape

        data_dropout_mask = F.dropout(
            mel_inputs.new_ones(batch_size, n_frame_per_step),
            p=self.hparams.data_dropout,
            training=dropout_flag,
        ).unsqueeze(2)
        mel_inputs = mel_inputs * data_dropout_mask
        return mel_inputs

    def get_dropout_while_eval(self, dropout_while_eval: bool) -> bool:
        r"""
        Returns the flag to be true or false based on the value given during evaluation

        Args:
            dropout_while_eval (bool): flag to dropout while eval or not

        Returns:
            (bool): dropout flag
        """
        if dropout_while_eval:
            dropout_flag = True
        else:
            dropout_flag = self.training

        return dropout_flag

    def get_absorption_state_scaling_factor(self, mel_inputs_lengths, log_alpha_scaled, text_lengths):
        r"""
        Returns the final scaling factor of absorption state
        Args:
            mel_inputs_lengths (torch.IntTensor): Input size of mels to
                    get the last timestep of log_alpha_scaled
            log_alpha_scaled (torch.FloatTEnsor): State probabilities
            text_lengths (torch.IntTensor): length of the states to
                    mask the values of states lengths
                (
                    Useful when the batch has very different lengths,
                    when the length of an observation is less than
                    the number of max states, then the log alpha after
                    the state value is filled with -infs. So we mask
                    those values so that it only consider the states
                    which are needed for that length
                )

        Returns:

        """
        max_text_len = log_alpha_scaled.shape[2]
        mask_tensor = log_alpha_scaled.new_zeros(max_text_len)
        state_lengths_mask = torch.arange(max_text_len, out=mask_tensor).expand(len(text_lengths), max_text_len) < (
            text_lengths
        ).unsqueeze(1)

        last_log_alpha_scaled_index = (
            (mel_inputs_lengths - 1).unsqueeze(-1).expand(-1, self.N).unsqueeze(1)
        )  # Batch X Hidden State Size
        last_log_alpha_scaled = torch.gather(log_alpha_scaled, 1, last_log_alpha_scaled_index).squeeze(1)
        last_log_alpha_scaled = last_log_alpha_scaled.masked_fill(~state_lengths_mask, -float("inf"))

        last_transition_vector = torch.gather(self.transition_vector, 1, last_log_alpha_scaled_index).squeeze(1)
        last_transition_probability = torch.sigmoid(last_transition_vector)
        log_probability_of_transitioning = log_clamped(last_transition_probability)

        last_transition_probability_index = (
            torch.arange(max_text_len, out=mask_tensor).expand(len(text_lengths), max_text_len)
        ) == (text_lengths - 1).unsqueeze(1)
        log_probability_of_transitioning = log_probability_of_transitioning.masked_fill(
            ~last_transition_probability_index, -float("inf")
        )

        final_log_c = last_log_alpha_scaled + log_probability_of_transitioning

        # Uncomment the line below if you get nan values because of low precision
        # in half precision training
        # final_log_c = final_log_c.clamp(min=torch.finfo(final_log_c.dtype).min)

        sum_final_log_c = torch.logsumexp(final_log_c, dim=1)
        return sum_final_log_c

    @torch.inference_mode()
    def sample(self, encoder_outputs, sampling_temp=1.0, T=None):
        r"""
        Samples an output from the parameter models

        Args:
            encoder_outputs (float tensor): (batch, text_len, encoder_embedding_dim)
            T (int): Max time to sample

        Returns:
            x (list[float]): Output Observations
            z (list[int]): Hidden states travelled
        """
        if not T:
            T = self.hparams.max_sampling_time

        self.N = encoder_outputs.shape[1] - 1
        if self.hparams.ar_order > 0:
            ar_mel_inputs = self.go_tokens.unsqueeze(0)
        else:
            raise ValueError(
                "ar_order should be greater than 0,  \
                it is an Autoregressive model"
            )

        z, x = [], []
        t = 0

        # Sample Initial state
        current_z_number = 0
        z.append(current_z_number)

        h_post_prenet, c_post_prenet = self.init_lstm_states(1, self.hparams.post_prenet_rnn_dim, ar_mel_inputs)

        input_parameter_values = []
        output_parameter_values = []
        quantile = 1
        prenet_dropout_flag = self.get_dropout_while_eval(self.hparams.prenet_dropout_while_eval)

        while True:
            if self.hparams.data_dropout_while_sampling:
                dropout_mask = F.dropout(
                    ar_mel_inputs.new_ones(ar_mel_inputs.shape[0], ar_mel_inputs.shape[1], 1),
                    p=self.hparams.data_dropout,
                )
                ar_mel_inputs = dropout_mask * ar_mel_inputs

            prenet_output = self.prenet(ar_mel_inputs.flatten(1).unsqueeze(0), prenet_dropout_flag)
            # will be 1 while sampling
            h_post_prenet, c_post_prenet = self.post_prenet_rnn(
                prenet_output.squeeze(0), (h_post_prenet, c_post_prenet)
            )

            z_t = encoder_outputs[:, current_z_number]
            mean, std, transition_vector = self.decoder(h_post_prenet, z_t.unsqueeze(0))

            transition_probability = torch.sigmoid(transition_vector.flatten())
            staying_probability = torch.sigmoid(-transition_vector.flatten())
            input_parameter_values.append([ar_mel_inputs, current_z_number])
            output_parameter_values.append([mean, std, transition_probability])

            x_t = self.emission_model.sample(mean, std, sampling_temp=sampling_temp)

            if self.hparams.predict_means:
                x_t = mean
            ar_mel_inputs = torch.cat((ar_mel_inputs, x_t), dim=1)[:, 1:]

            x.append(x_t.flatten())

            transition_matrix = torch.cat((staying_probability, transition_probability))
            quantile *= staying_probability

            if not self.hparams.deterministic_transition:
                switch = transition_matrix.multinomial(1)[0].item()
            else:
                switch = quantile < self.hparams.duration_quantile_threshold

            if switch:
                current_z_number += 1
                quantile = 1

            z.append(current_z_number)

            if (current_z_number == self.N) or (T and t == T - 1):
                break

            t += 1

        return torch.stack(x), z, input_parameter_values, output_parameter_values

    def _initialize_log_state_priors(self, text_embeddings):
        """Creates the log pi in forward algorithm.

        Args:
            text_embeddings (torch.FloatTensor): used to create the log pi
                    on current device
        """
        log_state_priors = text_embeddings.new_full([self.N], -float("inf"))
        log_state_priors[0] = 0.0
        return log_state_priors


@dataclass
class TransitionModel(nn.Module):
    """Transition Model of the HMM, it represents the probability of transitioning
        form current state to all other states"""

    staying_probability: torch.FloatTensor = None
    transition_probability: torch.FloatTensor = None

    def update_current_values(self, staying: torch.FloatTensor, transitioning: torch.FloatTensor):
        """
        Make reference of the staying and transitioning probabilities as instance
        parameters of class
        """
        self.staying_probability = staying
        self.transition_probability = transitioning

    def forward(self, log_alpha_scaled, transition_vector, state_lengths):
        r"""
        product of the past state with transitional probabilities in log space

        Args:
            log_alpha_scaled (torch.Tensor): Multiply previous timestep's alphas by
                        transition matrix (in log domain)
                - shape: (batch size, N)
            transition_vector (torch.tensor): transition vector for each state
                - shape: (N)
            state_lengths (int tensor): Lengths of states in a batch
                - shape: (batch)

        Returns:
            out (torch.FloatTensor): log probability of transitioning to each state
        """
        T_max = log_alpha_scaled.shape[1]

        transition_probability = torch.sigmoid(transition_vector)
        staying_probability = torch.sigmoid(-transition_vector)

        self.update_current_values(staying_probability, transition_probability)

        log_staying_probability = log_clamped(staying_probability)
        log_transition_probability = log_clamped(transition_probability)

        staying = log_alpha_scaled + log_staying_probability
        leaving = log_alpha_scaled + log_transition_probability
        leaving = leaving.roll(1, dims=1)
        leaving[:, 0] = -float("inf")

        mask_tensor = log_alpha_scaled.new_zeros(T_max)
        not_state_lengths_mask = ~(
            torch.arange(T_max, out=mask_tensor).expand(len(state_lengths), T_max) < (state_lengths).unsqueeze(1)
        )

        out = logsumexp(torch.stack((staying, leaving), dim=2), dim=2)

        out = out.masked_fill(not_state_lengths_mask, -float("inf"))

        return out


@dataclass
class EmissionModel(nn.Module):
    """Emission Model of the HMM, it represents the probability of
        emitting an observation based on the current state"""

    distribution_function: tdist.Distribution = tdist.normal.Normal

    def sample(self, means, stds, sampling_temp):
        return self.distribution_function(means, stds * sampling_temp).sample() if sampling_temp > 0 else means

    def forward(self, x_t, means, stds, state_lengths):
        r"""Calculates the log probability of the the given data (x_t)
            being observed from states with given means and stds
        Args:
            x_t (float tensor) : observation at current time step
                - shape: (batch, feature_dim)
            means (float tensor): means of the distributions of hidden states
                - shape: (batch, hidden_state, feature_dim)
            stds (float tensor): standard deviations of the distributions of the hidden states
                - shape: (batch, hidden_state, feature_dim)
            state_lengths (int tensor): Lengths of states in a batch
                - shape: (batch)

        Returns:
            out (float tensor): observation log likelihoods,
                                    expressing the probability of an observation
                being generated from a state i
                shape: (batch, hidden_state)
        """
        emission_dists = self.distribution_function(means, stds)
        out = emission_dists.log_prob(x_t.unsqueeze(1))
        state_lengths_mask = sequence_mask(state_lengths)
        out = torch.sum(out * state_lengths_mask, dim=2)
        return out