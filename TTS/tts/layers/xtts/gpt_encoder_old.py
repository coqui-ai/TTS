import functools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import deepspeed
    from deepspeed.ops.transformer.inference import DeepSpeedTransformerInferenceKernel
except ImportError:
    pass

import dlas.codes.torch_intermediary as ml
from dlas.codes.models.arch_util import AttentionBlock
from dlas.codes.trainer.networks import register_model
from dlas.codes.utils.transformers.stream_generator import init_stream_support
from dlas.codes.utils.util import opt_get
from transformers import GPT2Config, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

init_stream_support()


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
        do_checkpointing=False,
        mean=False,
    ):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=do_checkpointing))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h[:, :, 0]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    layers,
    model_dim,
    heads,
    max_mel_seq_len,
    max_text_seq_len,
    max_prompt_len,
    checkpointing,
):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    # def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    #     attn_output = torch.nn.functional.scaled_dot_product_attention(
    #         query, key, value, dropout_p=self.attn_dropout.p, is_causal=True
    #     )
    #     return attn_output, None

    # for i in range(len(gpt.h)):
    #     gpt.h[i].attn._attn = types.MethodType(
    #         _attn, gpt.h[i].attn
    #     )

    mel_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    # gpt = torch.compile(gpt, mode="reduce-overhead", fullgraph=True)
    return gpt, mel_pos_emb, text_pos_emb, None, None


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
            nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
            nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 16, channels // 2),
            nn.ReLU(),
            nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
            nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(channels // 8, channels),
            nn.ReLU(),
            nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
        )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        average_conditioning_embeddings=False,
        freeze_everything_but_position_embeddings=False,
        freeze_conditioning_encoder=False,
        tortoise_compat=True,
        label_smoothing=0.0,
    ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            average_conditioning_embeddings: Whether or not conditioning embeddings should be averaged, instead of fed piecewise into the model.
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.start_prompt_token = start_mel_token
        self.stop_prompt_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.mel_length_compression = mel_length_compression
        # self.conditioning_encoder = ConditioningEncoder(
        #     80, model_dim, num_attn_heads=heads
        # )
        self.average_conditioning_embeddings = average_conditioning_embeddings
        self.tortoise_compat = tortoise_compat  # credit to https://github.com/152334H/DL-Art-School/commit/ae80992817059acf6eef38a680efa5124cee570b
        # nn.Embedding
        self.text_embedding = ml.Embedding(self.number_text_tokens, model_dim)
        if use_mel_codes_as_input:
            # nn.Embedding
            self.mel_embedding = ml.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens,
            self.max_text_tokens,
            self.max_prompt_tokens,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = ml.Linear(model_dim, self.number_text_tokens)
        self.mel_head = ml.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

        if freeze_conditioning_encoder:
            print(" > Freezing conditioning encoder.")
            for p in self.conditioning_encoder.parameters():
                p.requires_grad = False
                p.DO_NOT_TRAIN = True

        if freeze_everything_but_position_embeddings:
            for p in self.parameters():
                p.requires_grad = False
                p.DO_NOT_TRAIN = True
            for m in [self.mel_pos_embedding, self.text_pos_embedding]:
                for p in m.parameters():
                    del p.DO_NOT_TRAIN
                    p.requires_grad = True

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning_encoder": list(self.conditioning_encoder.parameters()),
            "gpt": list(self.gpt.parameters()),
            "heads": list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }

    def post_init_gpt2_config(self, kv_cache=True, use_deepspeed=False, use_deepspeed_f16=False):
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

        if use_deepspeed:
            # init deepspeed inference engine
            if use_deepspeed_f16:
                self.gpt.wte = self.mel_embedding.half()
                self.gpt.wpe = self.mel_pos_embedding.half()
            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float16 if use_deepspeed_f16 else torch.float32,  # desired data type of output
                replace_method="auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True,  # replace the model with the kernel injector
            )
            self.inference_model = self.ds_engine.module.eval()

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(mel_lengths)):
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
        get_attns=False,
        return_latent=False,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        if prompt is not None and speech_conditioning_inputs is not None:
            offset = speech_conditioning_inputs.shape[1] + prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat(
                    [speech_conditioning_inputs, prompt, first_inputs, second_inputs],
                    dim=1,
                )
            else:
                emb = torch.cat([speech_conditioning_inputs, prompt, first_inputs], dim=1)
        elif speech_conditioning_inputs is not None:
            offset = speech_conditioning_inputs.shape[1]
            if second_inputs is not None:
                emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)
        elif prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_prompt = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_prompt, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input):
        speech_conditioning_input = (
            speech_conditioning_input.unsqueeze(1)
            if len(speech_conditioning_input.shape) == 3
            else speech_conditioning_input
        )
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        conds = conds.mean(dim=1)
        return conds

    def get_prompts(self, prompt_codes):
        """
        Create a prompt from the mel codes. This is used to condition the model on the mel codes.
        Pad the prompt with start and stop mel tokens.
        """
        prompt = prompt_codes
        if self.training:
            prompt_len = random.randint(1, 9)  # in secs
            prompt_len = prompt_len * 24  # in frames

            if prompt_codes.shape[1] < prompt_len:
                prompt_len = prompt_codes.shape[-1]
                start = 0
            else:
                start = random.randint(0, prompt_codes.shape[-1] - prompt_len)

            prompt = prompt_codes[:, start : start + prompt_len]

        # add start and stop tokens
        prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
        return prompt

    # def get_prompts(self, prompt_codes):
    #     """
    #     Create a prompt from the mel codes. This is used to condition the model on the mel codes.
    #     Pad the prompt with start and stop mel tokens.
    #     """
    #     prompt = prompt_codes
    #     if self.training:
    #         max_prompt_len = 9 * 24
    #         if prompt_codes.shape[1] < max_prompt_len:
    #             prompt = prompt_codes
    #         else:
    #             start = random.randint(0, prompt_codes.shape[1] - max_prompt_len)
    #             prompt = prompt_codes[:, start : start + max_prompt_len]

    #     # add start and stop tokens
    #     prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
    #     prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
    #     return prompt

    def forward(
        self,
        speech_conditioning_input,
        text_inputs,
        text_lengths,
        mel_codes,
        wav_lengths,
        prompt_codes,
        loss_weights=None,
        text_first=True,
        return_attentions=False,
        return_latent=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,80,s)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        # ❗ FIXIT
        speech_conditioning_input = None
        if self.max_conditioning_inputs == 0:
            assert (
                speech_conditioning_input is None
            ), " ❗ speech_conditioning_input is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        # Due to the convolution in DVAE, codes do not end with silence at the right place. Rather it predicts some intermediate values
        # Like [..., 186, 45, 45, 83] where actually it should end with 186.
        # We take last 3 codes to prevent abrupt ending of the audio.
        # TODO: This is might need some testing.
        mel_lengths = torch.ceil(wav_lengths / self.mel_length_compression).long() + 3

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = mel_lengths.max()

        if max_mel_len > mel_codes.shape[-1]:
            mel_codes = F.pad(mel_codes, (0, max_mel_len - mel_codes.shape[-1]))

        # mel_lengths[mel_lengths >= max_mel_len] = max_mel_len

        # silence aware lengths, skip the silence tokens at the end of the mel codes.
        silence = True
        for idx, l in enumerate(mel_lengths):
            length = l.item()
            while silence:
                if mel_codes[idx, length - 1] != 83:
                    break
                length -= 1
            mel_lengths[idx] = length

        # Lovely assertions
        assert (
            max_mel_len <= mel_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > mel_codes.shape[-1] ({mel_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        # Append silence token to mel codes
        mel_codes = F.pad(mel_codes[:, :max_mel_len], (0, 1), value=self.stop_mel_token)

        # Pad mel codes with STOP_MEL_TOKEN
        mel_codes = self.set_mel_padding(mel_codes, mel_lengths)

        # Compute speech conditioning input
        conds = None
        if speech_conditioning_input is not None:
            if not return_latent:
                # Compute speech conditioning input
                speech_conditioning_input = (
                    speech_conditioning_input.unsqueeze(1)
                    if len(speech_conditioning_input.shape) == 3
                    else speech_conditioning_input
                )

                conds = []
                for j in range(speech_conditioning_input.shape[1]):
                    conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
                conds = torch.stack(conds, dim=1)
                if self.average_conditioning_embeddings:
                    conds = conds.mean(dim=1).unsqueeze(1)
            else:
                # already computed
                conds = speech_conditioning_input.unsqueeze(1)

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )

        # Set attn_mask
        attn_mask_text = None
        attn_mask_mel = None
        if not return_latent:
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                mel_codes.shape[0],
                mel_codes.shape[1],
                dtype=torch.bool,
                device=mel_codes.device,
            )

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1 :] = 0.0

            for idx, l in enumerate(mel_lengths):
                attn_mask_mel[idx, l + 1 :] = 0.0

        # Compute text embeddings + positional embeddings
        # print(" > text input latent:", text_inputs)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_embedding(mel_codes) + self.mel_pos_embedding(mel_codes)

        # Compute prompt embeddings + positional embeddings
        prompt = self.get_prompts(prompt_codes)

        prompt_emb = self.mel_embedding(prompt).detach() + self.mel_pos_embedding(prompt).detach()

        # Get logits
        sub = -4  # don't ask me why 😄
        if self.training:
            sub = -1
        text_logits, mel_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=prompt_emb,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        if return_latent:
            return mel_logits[:, :sub]  # sub to prevent bla.

        if return_attentions:
            return mel_logits

        # Set paddings to -1 to ignore them in loss
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1 :] = -1

        for idx, l in enumerate(mel_lengths):
            mel_targets[idx, l + 1 :] = -1

        # check if stoptoken is in every row of mel_targets
        assert (mel_targets == self.stop_mel_token).sum() >= mel_targets.shape[
            0
        ], f" ❗ mel_targets does not contain stop token ({self.stop_mel_token}) in every row."

        # Compute losses
        loss_text = F.cross_entropy(
            text_logits, text_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )

        # if loss_weights is not None:
        #     loss_text = loss_text * loss_weights[:, None]
        #     loss_mel = loss_mel * loss_weights[:, None]
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def text_forward(self, speech_conditioning_input, text_inputs, text_lengths):
        """
        Performs autoregressive modeling on only text. Still requires a speech_conditioning_input due to the way the
        model inputs are formatted. Just provide any audio clip (arguably, zeros could be provided).
        """
        # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
        # chopping the inputs by the maximum actual length.
        max_text_len = text_lengths.max()
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        speech_conditioning_input = (
            speech_conditioning_input.unsqueeze(1)
            if len(speech_conditioning_input.shape) == 3
            else speech_conditioning_input
        )
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        if self.average_conditioning_embeddings:
            conds = conds.mean(dim=1).unsqueeze(1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs) + self.text_solo_embedding
        text_logits = self.get_logits(conds, text_emb, self.text_head)
        loss_text = F.cross_entropy(text_logits, text_targets.long())
        return loss_text.mean()

    def speech_forward(self, speech_conditioning_input, mel_codes, wav_lengths, raw_mels=None):
        """
        Performs autoregressive modeling on only speech data.
        """
        assert self.max_mel_tokens >= mel_codes.shape[1], f"{mel_codes.shape[1]}"

        # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
        # chopping the inputs by the maximum actual length.
        max_mel_len = wav_lengths.max() // self.mel_length_compression
        mel_codes = F.pad(mel_codes[:, :max_mel_len], (0, 1), value=self.stop_mel_token)
        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        if raw_mels is not None:
            raw_mels = raw_mels[:, :, : max_mel_len * 4]

        speech_conditioning_input = (
            speech_conditioning_input.unsqueeze(1)
            if len(speech_conditioning_input.shape) == 3
            else speech_conditioning_input
        )
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
        conds = torch.stack(conds, dim=1)
        if self.average_conditioning_embeddings:
            conds = conds.mean(dim=1).unsqueeze(1)

        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        if raw_mels is not None:
            mel_inp = F.pad(raw_mels, (0, 4))
        else:
            mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes) + self.mel_solo_embedding
        mel_logits = self.get_logits(conds, mel_emb, self.mel_head)
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long())
        return loss_mel.mean()

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.inference_model.generate_stream(
            fake_inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=self.max_mel_tokens * 2 + self.max_prompt_tokens + self.max_text_tokens,
            do_stream=True,
            **hf_generate_kwargs,
        )

    def compute_embeddings(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        prompt_codes=None,
        pad_input_text=False,
    ):
        if pad_input_text and text_inputs.shape[1] < 250:
            text_inputs = F.pad(text_inputs, (0, 250 - text_inputs.shape[1]), value=self.stop_text_token)
        else:
            text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)

        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        print(" > Text inputs:", text_inputs)
        if prompt_codes is not None:
            prompt_codes = self.get_prompts(prompt_codes)
            prompt_emb = self.mel_embedding(prompt_codes) + self.mel_pos_embedding(prompt_codes)
            print(" > Prompt inputs:", prompt_codes)
            print(" > Prompt inputs shape:", prompt_codes.shape)
            emb = torch.cat([prompt_emb, emb], dim=1)

        if speech_conditioning_latent is not None:
            conds = speech_conditioning_latent.unsqueeze(1)
            emb = torch.cat([conds, emb], dim=1)

        self.inference_model.store_prefix_emb(emb)

        fake_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        fake_inputs[:, -1] = self.start_mel_token

        if input_tokens is not None:
            fake_inputs = torch.cat([fake_inputs, input_tokens], dim=1)
        return fake_inputs

    def inference_speech(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        prompt_codes=None,
        pad_input_text=False,
        **hf_generate_kwargs,
    ):
        if pad_input_text and text_inputs.shape[1] < 250:
            text_inputs = F.pad(text_inputs, (0, 250 - text_inputs.shape[1]), value=self.stop_text_token)
        else:
            text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)

        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        print(" > Text inputs:", text_inputs)
        if prompt_codes is not None:
            prompt_codes = self.get_prompts(prompt_codes)
            prompt_emb = self.mel_embedding(prompt_codes) + self.mel_pos_embedding(prompt_codes)
            print(" > Prompt inputs:", prompt_codes)
            print(" > Prompt inputs shape:", prompt_codes.shape)
            emb = torch.cat([prompt_emb, emb], dim=1)

        if speech_conditioning_latent is not None:
            conds = speech_conditioning_latent.unsqueeze(1)
            emb = torch.cat([conds, emb], dim=1)

        self.inference_model.store_prefix_emb(emb)

        fake_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        fake_inputs[:, -1] = self.start_mel_token

        if input_tokens is not None:
            fake_inputs = torch.cat([fake_inputs, input_tokens], dim=1)

        gen = self.inference_model.generate(
            fake_inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=self.max_mel_tokens * 2 + self.max_prompt_tokens + self.max_text_tokens,
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, fake_inputs.shape[1] :], gen
        return gen[:, fake_inputs.shape[1] :]

    # Turns the (utterly insane) output of HF.generate() into a far more sane output:
    # [tensors(B,H,S,S)]. Outer=layers, B=batch,H=head,S=sequence
    def make_hf_generate_attentions_sane(self, attentions):
        layers = [[] for _ in range(len(attentions[0]))]
        full_attention_size = attentions[-1][0].shape[-1]
        for i, gen in enumerate(attentions):
            for j, lyr in enumerate(gen):
                layers[j].append(F.pad(lyr, (0, full_attention_size - lyr.shape[-1])))
        catted = []
        for lyr in layers:
            catted.append(torch.cat(lyr, dim=2))
        return catted

    def convert_attentions_to_aligned_codes(self, text, attentions, codes, num_conds):
        """
        This was an attempt to make some sense out of the attention matrix retrieved from the unified_voice model. Unfortunately, I can't use it for aligning text & voice.
        """
        text_padding = num_conds + 2
        num_text = text.shape[-1]
        num_context = num_text + text_padding
        assert num_context + 1 == attentions[0][0].shape[-1]
        attentions = self.make_hf_generate_attentions_sane(attentions)
        results = [torch.empty_like(codes) for _ in range(len(attentions))]
        for l, layer in enumerate(attentions):
            dec_context = layer[:, :, num_context:, :]
            # Mask out everything that isn't text (including the start token, which gets a LOT of attention)
            dec_context[:, :, :, : text_padding + 1] = 0
            dec_context[:, :, :, num_context:] = 0
            for h in range(dec_context.shape[1]):
                dec_context_indices = torch.argmax(dec_context[0, h], dim=-1)
                print(f"layer_{l};head_{h}: " + str(dec_context_indices))
        for t, att_tok in enumerate(attentions):
            combined_attention_weights = torch.zeros((codes.shape[0], num_text), device=codes.device)
            for lyr in att_tok:
                token_to_text_attentions = lyr[:, :, -1, text_padding : (text_padding + num_text)].sum(dim=1)
                combined_attention_weights = combined_attention_weights + token_to_text_attentions
                break
            most_attended_text_token = combined_attention_weights.argmax(dim=-1)
            results[:, t] = most_attended_text_token
        eos_token_mask = codes != self.stop_mel_token
        return results * eos_token_mask


@register_model
def register_unified_voice_prompt(opt_net, opt):
    return UnifiedVoice(**opt_get(opt_net, ["kwargs"], {}))


if __name__ == "__main__":
    gpt = UnifiedVoice(
        model_dim=256,
        heads=4,
        train_solo_embeddings=True,
        use_mel_codes_as_input=True,
        max_conditioning_inputs=4,
        freeze_everything_but_position_embeddings=True,
    )
    l = gpt(
        torch.randn(2, 3, 80, 800),
        torch.randint(high=256, size=(2, 120)),
        torch.tensor([32, 120]),
        torch.randint(high=8192, size=(2, 250)),
        torch.tensor([250 * 256, 195 * 256]),
    )
    # gpt.text_forward(torch.randn(2,80,800), torch.randint(high=50, size=(2,80)), torch.tensor([32, 80]))
