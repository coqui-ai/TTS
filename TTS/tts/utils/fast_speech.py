import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class DurationCalculator():
    def calculate_durations(self, att_ws, ilens, olens):
        """calculate duration from given alignment matrices"""
        durations = [self._calculate_duration(att_w, ilen, olen) for att_w, ilen, olen in zip(att_ws, ilens, olens)]
        return pad_sequence(durations, batch_first=True)

    @staticmethod
    def _calculate_duration(att_w, ilen, olen):
        '''
        attw : batch x outs x ins
        '''
        durations = torch.stack([att_w[:olen, :ilen].argmax(-1).eq(i).sum() for i in range(ilen)])
        return durations

    def calculate_scores(self, att_ws, ilens, olens):
        """calculate scores per duration step"""
        scores = [self._calculate_scores(att_w, ilen, olen, self.K) for att_w, ilen, olen in zip(att_ws, ilens, olens)]
        return pad_list(scores, 0)

    @staticmethod
    def _calculate_scores(att_w, ilen, olen, k):
        # which input is attended for each output
        scores = [None] * ilen
        values, idxs = att_w[:olen, :ilen].max(-1)
        for i in range(ilen):
            vals = values[torch.where(idxs == i)]
            scores[i] = vals
        scores = [torch.nn.functional.pad(score, (0, k - score.shape[0])) for score in scores]
        return torch.stack(scores)