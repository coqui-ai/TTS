import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.utils.ssim import ssim


# pylint: disable=abstract-method
# relates https://github.com/pytorch/pytorch/issues/42305
class L1LossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T X D
            target: B x T x D
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length,
                             max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask,
                                      target * mask,
                                      reduction='none')
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction='sum')
            loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super(MSELossMasked, self).__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T X D
            target: B x T x D
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length,
                             max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask,
                                       target * mask,
                                       reduction='none')
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask,
                                       target * mask,
                                       reduction='sum')
            loss = loss / mask.sum()
        return loss


class SSIMLoss(torch.nn.Module):
    """SSIM loss as explained here https://en.wikipedia.org/wiki/Structural_similarity"""
    def __init__(self):
        super().__init__()
        self.loss_func = ssim

    def forward(self, y_hat, y, length=None):
        """
        Args:
            y_hat (tensor): model prediction values.
            y (tensor): target values.
            length (tensor): length of each sample in a batch.
        Shapes:
            y_hat: B x T X D
            y: B x T x D
            length: B
         Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        if length is not None:
            m = sequence_mask(sequence_length=length,
                              max_len=y.size(1)).unsqueeze(2).float().to(
                                  y_hat.device)
            y_hat, y = y_hat * m, y * m
        return 1 - self.loss_func(y_hat.unsqueeze(1), y.unsqueeze(1))


class AttentionEntropyLoss(nn.Module):
    # pylint: disable=R0201
    def forward(self, align):
        """
        Forces attention to be more decisive by penalizing
        soft attention weights

        TODO: arguments
        TODO: unit_test
        """
        entropy = torch.distributions.Categorical(probs=align).entropy()
        loss = (entropy / np.log(align.shape[1])).mean()
        return loss


class BCELossMasked(nn.Module):
    def __init__(self, pos_weight):
        super(BCELossMasked, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        if length is not None:
            mask = sequence_mask(sequence_length=length,
                                 max_len=target.size(1)).float()
            x = x * mask
            target = target * mask
            num_items = mask.sum()
        else:
            num_items = torch.numel(x)
        loss = functional.binary_cross_entropy_with_logits(
            x,
            target,
            pos_weight=self.pos_weight,
            reduction='sum')
        loss = loss / num_items
        return loss


class DifferentailSpectralLoss(nn.Module):
    """Differential Spectral Loss
        https://arxiv.org/ftp/arxiv/papers/1909/1909.10302.pdf"""

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, x, target, length=None):
        """
         Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        x_diff = x[:, 1:] - x[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        if length is None:
            return self.loss_func(x_diff, target_diff)
        return self.loss_func(x_diff, target_diff, length-1)


class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(
                ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen).to(olen), torch.arange(ilen).to(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen)**2 /
                               (2 * (sigma**2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class Huber(nn.Module):
    # pylint: disable=R0201
    def forward(self, x, y, length=None):
        """
        Shapes:
            x: B x T
            y: B x T
            length: B
        """
        mask = sequence_mask(sequence_length=length, max_len=y.size(1)).float()
        return torch.nn.functional.smooth_l1_loss(
            x * mask, y * mask, reduction='sum') / mask.sum()


########################
# MODEL LOSS LAYERS
########################

class TacotronLoss(torch.nn.Module):
    """Collection of Tacotron set-up based on provided config."""
    def __init__(self, c, stopnet_pos_weight=10, ga_sigma=0.4):
        super(TacotronLoss, self).__init__()
        self.stopnet_pos_weight = stopnet_pos_weight
        self.ga_alpha = c.ga_alpha
        self.decoder_diff_spec_alpha = c.decoder_diff_spec_alpha
        self.postnet_diff_spec_alpha = c.postnet_diff_spec_alpha
        self.decoder_alpha = c.decoder_loss_alpha
        self.postnet_alpha = c.postnet_loss_alpha
        self.decoder_ssim_alpha = c.decoder_ssim_alpha
        self.postnet_ssim_alpha = c.postnet_ssim_alpha
        self.config = c

        # postnet and decoder loss
        if c.loss_masking:
            self.criterion = L1LossMasked(c.seq_len_norm) if c.model in [
                "Tacotron"
            ] else MSELossMasked(c.seq_len_norm)
        else:
            self.criterion = nn.L1Loss() if c.model in ["Tacotron"
                                                        ] else nn.MSELoss()
        # guided attention loss
        if c.ga_alpha > 0:
            self.criterion_ga = GuidedAttentionLoss(sigma=ga_sigma)
        # differential spectral loss
        if c.postnet_diff_spec_alpha > 0 or c.decoder_diff_spec_alpha > 0:
            self.criterion_diff_spec = DifferentailSpectralLoss(loss_func=self.criterion)
        # ssim loss
        if c.postnet_ssim_alpha > 0 or c.decoder_ssim_alpha > 0:
            self.criterion_ssim = SSIMLoss()
        # stopnet loss
        # pylint: disable=not-callable
        self.criterion_st = BCELossMasked(
            pos_weight=torch.tensor(stopnet_pos_weight)) if c.stopnet else None

    def forward(self, postnet_output, decoder_output, mel_input, linear_input,
                stopnet_output, stopnet_target, output_lens, decoder_b_output,
                alignments, alignment_lens, alignments_backwards, input_lens):


        # decoder outputs linear or mel spectrograms for Tacotron and Tacotron2
        # the target should be set acccordingly
        postnet_target = linear_input if self.config.model.lower() in ["tacotron"] else mel_input

        return_dict = {}
        # remove lengths if no masking is applied
        if not self.config.loss_masking:
            output_lens = None
        # decoder and postnet losses
        if self.config.loss_masking:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input,
                                              output_lens)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target,
                                              output_lens)
        else:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target)
        loss = self.decoder_alpha * decoder_loss + self.postnet_alpha * postnet_loss
        return_dict['decoder_loss'] = decoder_loss
        return_dict['postnet_loss'] = postnet_loss

        # stopnet loss
        stop_loss = self.criterion_st(
            stopnet_output, stopnet_target,
            output_lens) if self.config.stopnet else torch.zeros(1)
        if not self.config.separate_stopnet and self.config.stopnet:
            loss += stop_loss
        return_dict['stopnet_loss'] = stop_loss

        # backward decoder loss (if enabled)
        if self.config.bidirectional_decoder:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(
                    torch.flip(decoder_b_output, dims=(1, )), mel_input,
                    output_lens)
            else:
                decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1, )), mel_input)
            decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_b_output, dims=(1, )), decoder_output)
            loss += self.decoder_alpha * (decoder_b_loss + decoder_c_loss)
            return_dict['decoder_b_loss'] = decoder_b_loss
            return_dict['decoder_c_loss'] = decoder_c_loss

        # double decoder consistency loss (if enabled)
        if self.config.double_decoder_consistency:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input,
                                                output_lens)
            else:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input)
            # decoder_c_loss = torch.nn.functional.l1_loss(decoder_b_output, decoder_output)
            attention_c_loss = torch.nn.functional.l1_loss(alignments, alignments_backwards)
            loss += self.decoder_alpha * (decoder_b_loss + attention_c_loss)
            return_dict['decoder_coarse_loss'] = decoder_b_loss
            return_dict['decoder_ddc_loss'] = attention_c_loss

        # guided attention loss (if enabled)
        if self.config.ga_alpha > 0:
            ga_loss = self.criterion_ga(alignments, input_lens, alignment_lens)
            loss += ga_loss * self.ga_alpha
            return_dict['ga_loss'] = ga_loss

        # decoder differential spectral loss
        if self.config.decoder_diff_spec_alpha > 0:
            decoder_diff_spec_loss = self.criterion_diff_spec(decoder_output, mel_input, output_lens)
            loss += decoder_diff_spec_loss * self.decoder_diff_spec_alpha
            return_dict['decoder_diff_spec_loss'] = decoder_diff_spec_loss

        # postnet differential spectral loss
        if self.config.postnet_diff_spec_alpha > 0:
            postnet_diff_spec_loss = self.criterion_diff_spec(postnet_output, postnet_target, output_lens)
            loss += postnet_diff_spec_loss * self.postnet_diff_spec_alpha
            return_dict['postnet_diff_spec_loss'] = postnet_diff_spec_loss

        # decoder ssim loss
        if self.config.decoder_ssim_alpha > 0:
            decoder_ssim_loss = self.criterion_ssim(decoder_output, mel_input, output_lens)
            loss += decoder_ssim_loss * self.postnet_ssim_alpha
            return_dict['decoder_ssim_loss'] = decoder_ssim_loss

        # postnet ssim loss
        if self.config.postnet_ssim_alpha > 0:
            postnet_ssim_loss = self.criterion_ssim(postnet_output, postnet_target, output_lens)
            loss += postnet_ssim_loss * self.postnet_ssim_alpha
            return_dict['postnet_ssim_loss'] = postnet_ssim_loss

        return_dict['loss'] = loss

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict


class GlowTTSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant_factor = 0.5 * math.log(2 * math.pi)

    def forward(self, z, means, scales, log_det, y_lengths, o_dur_log,
                o_attn_dur, x_lengths):
        return_dict = {}
        # flow loss - neg log likelihood
        pz = torch.sum(scales) + 0.5 * torch.sum(
            torch.exp(-2 * scales) * (z - means)**2)
        log_mle = self.constant_factor + (pz - torch.sum(log_det)) / (
            torch.sum(y_lengths) * z.shape[1])
        # duration loss - MSE
        # loss_dur = torch.sum((o_dur_log - o_attn_dur)**2) / torch.sum(x_lengths)
        # duration loss - huber loss
        loss_dur = torch.nn.functional.smooth_l1_loss(
            o_dur_log, o_attn_dur, reduction='sum') / torch.sum(x_lengths)
        return_dict['loss'] = log_mle + loss_dur
        return_dict['log_mle'] = log_mle
        return_dict['loss_dur'] = loss_dur

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict


class SpeedySpeechLoss(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.l1 = L1LossMasked(False)
        self.ssim = SSIMLoss()
        self.huber = Huber()

        self.ssim_alpha = c.ssim_alpha
        self.huber_alpha = c.huber_alpha
        self.l1_alpha = c.l1_alpha

    def forward(self, decoder_output, decoder_target, decoder_output_lens, dur_output, dur_target, input_lens):
        l1_loss = self.l1(decoder_output, decoder_target, decoder_output_lens)
        ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
        huber_loss = self.huber(dur_output, dur_target, input_lens)
        loss = self.l1_alpha * l1_loss + self.ssim_alpha * ssim_loss + self.huber_alpha * huber_loss
        return {'loss': loss, 'loss_l1': l1_loss, 'loss_ssim': ssim_loss, 'loss_dur': huber_loss}


def mse_loss_custom(x, y):
    """MSE loss using the torch back-end without reduction.
    It uses less VRAM than the raw code"""
    expanded_x, expanded_y = torch.broadcast_tensors(x, y)
    return torch._C._nn.mse_loss(expanded_x, expanded_y, 0)  # pylint: disable=protected-access, c-extension-no-member


class MDNLoss(nn.Module):
    """Mixture of Density Network Loss as described in https://arxiv.org/pdf/2003.01950.pdf.
    """

    def forward(self, logp, text_lengths, mel_lengths):  # pylint: disable=no-self-use
        '''
        Shapes:
            mu: [B, D, T]
            log_sigma: [B, D, T]
            mel_spec: [B, D, T]
        '''
        B, T_seq, T_mel = logp.shape
        log_alpha = logp.new_ones(B, T_seq, T_mel)*(-1e4)
        log_alpha[:, 0, 0] = logp[:, 0, 0]
        for t in range(1, T_mel):
            prev_step = torch.cat([log_alpha[:, :, t-1:t], functional.pad(log_alpha[:, :, t-1:t],
                                                                          (0, 0, 1, -1), value=-1e4)], dim=-1)
            log_alpha[:, :, t] = torch.logsumexp(prev_step + 1e-4, dim=-1) + logp[:, :, t]
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1]
        mdn_loss = -alpha_last.mean() / T_seq
        return mdn_loss#, log_prob_matrix


class AlignTTSLoss(nn.Module):
    """Modified AlignTTS Loss.
    Computes following losses
        - L1 and SSIM losses from output spectrograms.
        - Huber loss for duration predictor.
        - MDNLoss for Mixture of Density Network.

    All the losses are aggregated by a weighted sum with the loss alphas.
    Alphas can be scheduled based on number of steps.

    Args:
        c (dict): TTS model configuration.
    """
    def __init__(self, c):
        super().__init__()
        self.mdn_loss = MDNLoss()
        self.spec_loss = MSELossMasked(False)
        self.ssim = SSIMLoss()
        self.dur_loss = MSELossMasked(False)

        self.ssim_alpha = c.ssim_alpha
        self.dur_loss_alpha = c.dur_loss_alpha
        self.spec_loss_alpha = c.spec_loss_alpha
        self.mdn_alpha = c.mdn_alpha

    def forward(self, logp, decoder_output, decoder_target, decoder_output_lens, dur_output, dur_target,
                input_lens, step, phase):
        ssim_alpha, dur_loss_alpha, spec_loss_alpha, mdn_alpha = self.set_alphas(
            step)
        spec_loss, ssim_loss, dur_loss, mdn_loss = 0, 0, 0, 0
        if phase == 0:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
        elif phase == 1:
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
        elif phase == 2:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
            spec_loss = self.spec_lossX(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
        elif phase == 3:
            dur_loss = self.dur_loss(dur_output.unsqueeze(2), dur_target.unsqueeze(2), input_lens)
        else:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
            dur_loss = self.dur_loss(dur_output.unsqueeze(2), dur_target.unsqueeze(2), input_lens)
        loss = spec_loss_alpha * spec_loss + ssim_alpha * ssim_loss + dur_loss_alpha * dur_loss + mdn_alpha * mdn_loss
        return {'loss': loss, 'loss_l1': spec_loss, 'loss_ssim': ssim_loss, 'loss_dur': dur_loss, 'mdn_loss': mdn_loss}

    @staticmethod
    def _set_alpha(step, alpha_settings):
        '''Set the loss alpha wrt number of steps.
        Return the corresponding value if no schedule is set.

        Example:
            Setting a alpha schedule.
            if ```alpha_settings``` is ```[[0, 1], [10000, 0.1]]```  then ```return_alpha == 1``` until 10k steps, then set to 0.1.
            if ```alpha_settings``` is a constant value then ```return_alpha``` is set to that constant.

        Args:
            step (int): number of training steps.
            alpha_settings (int or list): constant alpha value or a list defining the schedule as explained above.
        '''
        return_alpha = None
        if isinstance(alpha_settings, list):
            for key, alpha in alpha_settings:
                if key < step:
                    return_alpha = alpha
        elif isinstance(alpha_settings, (float, int)):
            return_alpha = alpha_settings
        return return_alpha

    def set_alphas(self, step):
        '''Set the alpha values for all the loss functions
        '''
        ssim_alpha = self._set_alpha(step, self.ssim_alpha)
        dur_loss_alpha = self._set_alpha(step, self.dur_loss_alpha)
        spec_loss_alpha = self._set_alpha(step, self.spec_loss_alpha)
        mdn_alpha = self._set_alpha(step, self.mdn_alpha)
        return ssim_alpha, dur_loss_alpha, spec_loss_alpha, mdn_alpha
