import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


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


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_channels, init_std=0.02, relative=False):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_channels)
        nn.init.normal_(self.emb.weight, mean=0.0, std=init_std)
        self.relative = relative

    def forward(self, x):
        seq_len = x.shape[1]
        if self.relative:
            start = torch.randint(seq_len, (1,), device=x.device).item()
            positions = torch.arange(start, start + seq_len, device=x.device)
        else:
            positions = torch.arange(seq_len, device=x.device)
        return self.emb(positions)

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def init_gpt(layers, model_channels, heads, max_mel_seq_len, max_text_seq_len, max_prompt_len, checkpointing):
    """
    Initializes a GPT-2 model and its position embeddings for a text-to-speech system.

    Args:
        layers (int): Number of layers in the GPT-2 model.
        model_channels (int): Dimension of the GPT-2 model.
        heads (int): Number of heads in the GPT-2 model.
        max_mel_seq_len (int): Maximum sequence length for the mel spectrogram.
        max_text_seq_len (int): Maximum sequence length for the text.
        max_prompt_len (int): Maximum length of the prompt.
        checkpointing (bool): Whether to use gradient checkpointing.

    Returns:
        gpt (GPT2Model): GPT-2 model.
        mel_pos_emb (LearnedPositionEmbeddings): Position embeddings for the mel spectrogram.
        text_pos_emb (LearnedPositionEmbeddings): Position embeddings for the text.
    """
    gpt_config = GPT2Config(
        vocab_size=123,
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_channels,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)

    del gpt.wpe
    del gpt.wte

    gpt.wpe = functools.partial(null_position_embeddings, dim=model_channels)

    audio_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_channels)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_channels)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_channels)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_channels)
    )

    return gpt, audio_pos_emb, text_pos_emb


class XTTSGPTEncoder(nn.Module):
    """XTTS GPT Encoder model implementation.
    Args:
        start_text_token (int): Index of the start token in the text vocabulary.
        stop_text_token (int): Index of the stop token in the text vocabulary.
        n_layers (int): Number of layers in the GPT-2 model.
        n_model_channels (int): Dimension of the GPT-2 model.
        n_heads (int): Number of heads in the GPT-2 model.
        max_text_tokens (int): Maximum number of text tokens.
        max_audio_tokens (int): Maximum number of audio tokens.
        max_prompt_tokens (int): Maximum number of prompt tokens.
        audio_len_compression (int): Compression factor for the audio length.
        number_text_tokens (int): Number of text tokens.
        number_audio_codes (int): Number of audio codes.
        start_mel_token (int): Index of the start token in the mel code vocabulary.
        stop_mel_token (int): Index of the stop token in the mel code vocabulary.
        checkpointing (bool): Whether or not to use gradient checkpointing at training.
    """

    _inference_flag = False

    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        n_layers=8,
        n_model_channels=512,
        n_heads=8,
        max_text_tokens=120,
        max_audio_tokens=250,
        max_prompt_tokens=70,
        audio_len_compression=1024,
        number_text_tokens=256,
        number_audio_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        checkpointing=True,
        label_smoothing=0.0,
    ):
        super().__init__()

        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_audio_codes = number_audio_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.start_prompt_token = start_mel_token
        self.stop_prompt_token = stop_mel_token
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model_channels = n_model_channels
        self.max_audio_tokens = -1 if max_audio_tokens == -1 else max_audio_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.audio_len_compression = audio_len_compression

        # embedding layers
        self.text_embedding = nn.Embedding(self.number_text_tokens, n_model_channels)
        self.audio_embedding = nn.Embedding(self.number_audio_codes, n_model_channels)
        self.prompt_embedding = nn.Embedding(self.number_audio_codes, n_model_channels)
        self.prompt_pos_embedding = LearnedPositionEmbeddings(24 * 9, n_model_channels)

        # initialize the GPT-2 model
        (
            self.gpt,
            self.audio_pos_embedding,
            self.text_pos_embedding,
        ) = init_gpt(
            n_layers,
            n_model_channels,
            n_heads,
            self.max_audio_tokens,
            self.max_text_tokens,
            self.max_prompt_tokens,
            checkpointing,
        )

        # output layers
        self.final_norm = nn.LayerNorm(n_model_channels)
        self.text_head = nn.Linear(n_model_channels, self.number_text_tokens)
        self.mel_head = nn.Linear(n_model_channels, self.number_audio_codes)

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning_encoder": list(self.conditioning_encoder.parameters()),
            "gpt": list(self.gpt.parameters()),
            "heads": list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }

    def init_model_for_inference(self, kv_cache=True, use_deepspeed=False, use_deepspeed_f16=False):
        self._inference_flag = True
        seq_length = self.max_prompt_tokens + self.max_audio_tokens + self.max_text_tokens
        gpt_config = GPT2Config(
            vocab_size=self.max_audio_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.n_model_channels,
            n_layer=self.n_layers,
            n_head=self.n_heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.audio_pos_embedding,
            self.audio_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        self.gpt.wte = self.audio_embedding

    def set_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_audio_tokens_padding(self, audio_tokens, audio_token_lens):
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(audio_token_lens)):
            actual_end = audio_token_lens[b]
            if actual_end < audio_tokens.shape[-1]:
                audio_tokens[b, actual_end:] = self.stop_mel_token
        return audio_tokens

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
        prompt = F.pad(prompt_codes, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt_codes, (0, 1), value=self.stop_prompt_token)
        return prompt

    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        prompt_codes,
        return_attentions=False,
        return_latent=False,
    ):
        max_text_len = text_lengths.max()

        # Due to the convolution in DVAE, codes do not end with silence at the right place. Rather it predicts some intermediate values
        # Like [..., 186, 45, 45, 83] where actually it should end with 186.
        # We take last 3 codes to prevent abrupt ending of the audio.
        # TODO: This is might need some testing.
        mel_lengths = torch.ceil(wav_lengths / self.mel_length_compression).long() + 3

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = mel_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        # silence aware lengths, skip the silence tokens at the end of the mel codes.
        silence = True
        for idx, l in enumerate(mel_lengths):
            length = l.item()
            while silence:
                if audio_codes[idx, length - 1] != 83:
                    break
                length -= 1
            mel_lengths[idx] = length

        # Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_mel_token)

        # Pad mel codes with STOP_MEL_TOKEN
        audio_codes = self.set_mel_padding(audio_codes, mel_lengths)

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
        text_inputs, _ = self.set_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        audio_codes, _ = self.set_inputs_and_targets(audio_codes, self.start_mel_token, self.stop_mel_token)

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
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1 :] = 0.0

            for idx, l in enumerate(mel_lengths):
                attn_mask_mel[idx, l + 1 :] = 0.0

        # Compute text embeddings + positional embeddings
        # print(" > text input latent:", text_inputs)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # Compute mel embeddings + positional embeddings
        audio_emb = self.audio_embedding(audio_codes) + self.audio_embedding(audio_codes)

        # Compute prompt embeddings + positional embeddings
        prompt = self.get_prompts(prompt_codes)

        # prompt_emb = self.audio_embedding(prompt).detach() + self.mel_pos_embedding(prompt).detach()
        prompt_emb = self.prompt_embedding(prompt) + self.prompt_pos_embedding(prompt)

        # dropout prompt embeddings
        prompt_emb = F.dropout(prompt_emb, p=0.1, training=self.training)

        # Get logits
        sub = -4  # don't ask me why 😄
        if self.training:
            sub = -1
        _, audio_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            audio_emb,
            self.mel_head,
            prompt=prompt_emb,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        return audio_logits[:, :sub]  # sub to prevent bla.

    def compute_embeddings(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        prompt_codes=None,
        pad_input_text=False,
    ):
        """Compute all the embeddings needed for inference."""
        if pad_input_text and text_inputs.shape[1] < 250:
            text_inputs = F.pad(text_inputs, (0, 250 - text_inputs.shape[1]), value=self.stop_text_token)
        else:
            text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)

        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        print(" > Text inputs:", text_inputs)
        if prompt_codes is not None:
            prompt_codes = self.get_prompts(prompt_codes)
            # prompt_emb = self.audio_embedding(prompt_codes) + self.mel_pos_embedding(prompt_codes)
            prompt_emb = self.prompt_embedding(prompt_codes) + self.prompt_pos_embedding(prompt_codes)

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

    def inference(
        self,
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

        if prompt_codes is not None:
            prompt_codes = self.get_prompts(prompt_codes)
            prompt_emb = self.prompt_embedding(prompt_codes) + self.prompt_pos_embedding(prompt_codes)
            emb = torch.cat([prompt_emb, emb], dim=1)

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
            max_length=self.max_audio_tokens * 2 + self.max_prompt_tokens + self.max_text_tokens,
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, fake_inputs.shape[1] :], gen
        return gen[:, fake_inputs.shape[1] :]
