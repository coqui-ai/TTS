from dataclasses import dataclass
from typing import Union

import jiwer
import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as f

from TTS.stt.datasets.tokenizer import Tokenizer
from TTS.stt.models.base_stt import BaseSTT
from TTS.tts.utils.helpers import sequence_mask


def compute_wer(logits, targets, logits_lenghts, target_lengths, tokenizer):
    pred_ids = np.argmax(logits, axis=-1)

    pred_texts = []
    for idx, pi in enumerate(pred_ids):
        pi = pi[: logits_lenghts[idx]]
        tokens = tokenizer.decode(pi)
        text = tokenizer.tokens_to_string(tokens)
        pred_texts.append("".join(text))

    label_texts = []
    for idx, pi in enumerate(targets):
        label = "".join(tokenizer.decode(pi))
        label = label[: target_lengths[idx]]
        label = label.replace(tokenizer.word_del_token, " ")
        label_texts.append(label)

    wer = jiwer.wer(label_texts, pred_texts)
    return wer, label_texts, pred_texts


@dataclass
class DeepSpeechArgs(Coqpit):
    n_tokens: int = 0  # The number of characters and +1 for CTC blank label
    n_context: int = (
        9  # The number of frames in the context - translates to the windows size of 2 * n_context + n_input.
    )
    context_step: int = 1  # The number of steps take between each context window.
    n_input: int = 26  # Number of MFCC features - TODO: Determine this programmatically from the sample rate
    n_hidden_1: int = 2048
    dropout_1: float = 0.1
    n_hidden_2: int = 2048
    dropout_2: float = 0.1
    n_hidden_3: int = 2048  # The number of units in the third layer, which feeds in to the LSTM
    dropout_3: float = 0.1
    n_hidden_4: int = 2048
    n_hidden_5: int = 2048
    dropout_5: float = 0.1
    layer_norm: bool = False  # TODO: add layer norm layers to the model
    relu_clip: float = 20.0


class ClippedReLU(nn.Module):
    def __init__(self, clip=20.0):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        return torch.clip(f.relu(x), max=self.clip)


class DeepSpeech(BaseSTT):
    def __init__(self, config: Union["DeepSpeechConfig", DeepSpeechArgs]):
        super().__init__(config)

        if hasattr(self, "config"):
            self.tokenizer = Tokenizer(vocab_dict=self.config.vocabulary)

        self.layer_1 = nn.Linear((self.args.n_context * 2 + 1) * self.args.n_input, self.args.n_hidden_1)
        self.dropout_1 = nn.Dropout(p=self.args.dropout_1)
        self.relu_1 = ClippedReLU(self.args.relu_clip)
        self.layer_2 = nn.Linear(self.args.n_hidden_1, self.args.n_hidden_2)
        self.dropout_2 = nn.Dropout(p=self.args.dropout_2)
        self.relu_2 = ClippedReLU(self.args.relu_clip)
        self.layer_3 = nn.Linear(self.args.n_hidden_2, self.args.n_hidden_3)
        self.dropout_3 = nn.Dropout(p=self.args.dropout_3)
        self.relu_3 = ClippedReLU(self.args.relu_clip)
        self.lstm = nn.LSTM(self.args.n_hidden_3, self.args.n_hidden_4, num_layers=1, batch_first=True)
        self.layer_5 = nn.Linear(self.args.n_hidden_4, self.args.n_hidden_5)
        self.dropout_5 = nn.Dropout(p=self.args.dropout_5)
        self.relu_5 = ClippedReLU(self.args.relu_clip)
        self.layer_6 = nn.Linear(self.args.n_hidden_5, self.args.n_tokens)

    @staticmethod
    def logits_to_text(logits, tokenizer):
        pred_ids = np.argmax(logits, axis=-1)

        pred_texts = []
        for pi in pred_ids:
            pred_texts.append("".join(tokenizer.decode(pi)))
        return pred_texts

    def reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Format the input tensor to define the context windows.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: formatted tensor with the last dimension is the context window.

        Shapes:
            x : :math:`[B, T, C]`
            Tensor: :math:`[B, T/context_step, C*2*n_context+C]`
        """
        # TODO: might need to pad x before unfold for the last window
        x = x.unfold(1, 2 * self.args.n_context + 1, self.args.context_step).permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], -1, self.args.n_input * 2 * self.args.n_context + self.args.n_input)
        return x

    def recompute_input_lengths(self, x):
        return ((x - (2 * self.args.n_context + 1)) / self.args.context_step) + 1

    def forward(self, x, previous_states: tuple[torch.Tensor, torch.Tensor]):
        x = self.layer_1(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        x = self.dropout_3(x)
        x = self.relu_3(x)
        new_state_h, new_state_c = self.lstm(x, previous_states)
        x = self.layer_5(new_state_h)
        x = self.dropout_5(x)
        x = self.relu_5(x)
        x = self.layer_6(x)
        x = x.log_softmax(dim=-1)
        return {"model_outputs": x, "model_states": (new_state_h, new_state_c)}

    @torch.no_grad()
    def inference(self, x, aux_inputs={"previous_states": None}):
        return self.forward(x, aux_inputs["previous_states"])

    def train_step(self, batch, criterion):
        outputs = self.forward(batch["features"], None)

        if self.config.loss_masking:
            output_mask = sequence_mask(batch["feature_lengths"])
            outputs["model_outputs"] = outputs["model_outputs"].masked_fill(
                ~output_mask.unsqueeze(-1), self.tokenizer.pad_token_id
            )
            target_mask = sequence_mask(batch["token_lengths"])
            target_lengths = target_mask.sum(-1)
            flattened_targets = batch["tokens"].masked_select(target_mask)

        # with autocast(enabled=False):  # avoid mixed_precision in criterion
        with torch.backends.cudnn.flags(enabled=False):
            loss = criterion(
                outputs["model_outputs"].transpose(0, 1),  # [B, T, C] -> [T, B, C]
                flattened_targets,
                batch["feature_lengths"],
                target_lengths,
            )

        wer, target_texts, pred_texts = compute_wer(
            outputs["model_outputs"].detach().cpu().numpy(),
            batch["tokens"].cpu().numpy(),
            batch["feature_lengths"].cpu().numpy(),
            batch["token_lengths"].cpu().numpy(),
            self.tokenizer,
        )

        # for visualization
        outputs["target_texts"] = target_texts
        outputs["pred_texts"] = pred_texts

        loss_dict = {"loss": loss, "wer": wer}
        return outputs, loss_dict

    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)

    def train_log(self, batch, outputs, dashboard_logger, assets, step, prefix="Train - "):
        pred_log_text = "\n".join([f"{idx} - {pred}" for idx, pred in enumerate(outputs["pred_texts"])])
        target_log_text = "\n".join([f"{idx} - {pred}" for idx, pred in enumerate(outputs["target_texts"])])

        dashboard_logger.add_text(f"{prefix} - Last Batch Predictions:", f"<pre>{pred_log_text}</pre>", step)
        dashboard_logger.add_text(f"{prefix} - Last Batch Target:", f"<pre>{target_log_text}</pre>", step)

    def eval_log(self, batch, outputs, dashboard_logger, assets, step):
        self.train_log(batch, outputs, dashboard_logger, assets, step, prefix="Eval - ")

    def get_criterion(self):
        ctc_loss = nn.CTCLoss(blank=self.config.vocabulary["[PAD]"], reduction="mean")
        return ctc_loss

    def format_batch(self, batch):
        batch["features"] = self.reshape_input(batch["features"])
        batch["feature_lengths"] = self.recompute_input_lengths(batch["feature_lengths"]).long()
        assert (
            batch["feature_lengths"].max() == batch["features"].shape[1]
        ), f"{batch['feature_lengths'].max()} vs {batch['features'].shape[1]}"
        return batch

    def load_checkpoint(self, config: Coqpit, checkpoint_path: str, eval: bool = False) -> None:
        return None


if __name__ == "__main__":

    args = DeepSpeechArgs(n_input=23, n_tokens=13)
    model = DeepSpeech(args)
    input_tensor = torch.rand(2, 57, 23)
    input_tensor = model.reshape_input(input_tensor)
    outputs = model.forward(input_tensor, None)
    print(outputs["model_outputs"].shape)
