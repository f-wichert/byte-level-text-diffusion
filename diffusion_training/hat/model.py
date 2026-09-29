import itertools
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import ModelOutput

from .config import (
    CrossAttentionConfig,
    DecoderHATModelConfig,
    EncoderHATModelConfig,
    HATArchitectureConfig,
    TransformerHATModelConfig,
)
from .splitter import HATSplitter
from .norm import RMSNorm
from .transformer_backbone import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)



def sample_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)[:, -1]


LLAMA_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


class HATCache:
    encoder_cache: DynamicCache
    backbone_cache: DynamicCache
    decoder_cache: DynamicCache

    def __init__(self):
        self.encoder_cache = DynamicCache()
        self.backbone_cache = DynamicCache()
        self.decoder_cache = DynamicCache()

    def get_backbone_cache(self) -> DynamicCache:
        return self.backbone_cache

    def get_decoder_cache(self) -> DynamicCache:
        return self.decoder_cache

    def get_encoder_cache(self) -> DynamicCache:
        return self.encoder_cache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, q_cos=None, q_sin=None, k_cos=None, k_sin=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    and allows for different sequence lengths.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        q_cos (`torch.Tensor`): The cosine part of the rotary embedding.
        q_sin (`torch.Tensor`): The sine part of the rotary embedding.
        k_cos (`torch.Tensor`): The cosine part of the rotary embedding.
        k_sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze
            cos[position_ids] and sin[position_ids] so that they can be properly
            broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape
            [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting
            unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids]
            broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key
        tensors rotated using the Rotary Position Embedding.
    """

    q_cos = q_cos.unsqueeze(unsqueeze_dim)
    q_sin = q_sin.unsqueeze(unsqueeze_dim)
    k_cos = k_cos.unsqueeze(unsqueeze_dim)
    k_sin = k_sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)

    return q_embed, k_embed


class HATBackbone(nn.Module):
    def __init__(self, config: TransformerHATModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = False,
    ) -> BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for backbone_layer in self.layers:
            layer_outputs = backbone_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return CausalLMOutputWithPast(
            hidden_states=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class HATDecoderConnector(nn.Module):
    def __init__(self, backbone_hiden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_word_embedding = torch.nn.Parameter(
            torch.empty(
                1,
                1,
                backbone_hiden_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )

    def forward(
        self,
        backbone_activations: torch.Tensor,
    ):
        activations = backbone_activations.clone()
        activations[:, -1:, :] = self.first_word_embedding
        activations = torch.roll(activations, shifts=1, dims=1)
        return activations


class HATDecoderBlock(nn.Module):
    def __init__(
        self,
        add_cross_attention: bool,
        config: DecoderHATModelConfig,
        layer_idx: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_cross_attention = add_cross_attention
        self.config = config
        self.llama_layer = LlamaDecoderLayer(config, layer_idx)
        self.llama_layer.self_attn.sliding_window = config.sliding_window
        if add_cross_attention:
            self.cross_attention = HATCrossAttention(
                hidden_size=config.cross_attention_config.hidden_size,
                hidden_size_kv=config.cross_attention_config.hidden_size_kv,
                hidden_size_q=config.cross_attention_config.hidden_size_q,
                config=config,
                cross_attention_config=config.cross_attention_config,
            )

        self.query_norm = RMSNorm(
            config.cross_attention_config.hidden_size_q,
            eps=config.rms_norm_eps,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            norm_in_fp32=False,
        )

        self.kv_norm = RMSNorm(
            config.cross_attention_config.hidden_size_kv,
            eps=config.rms_norm_eps,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            norm_in_fp32=False,
        )

    def apply_norm(self, activations):
        return self.query_norm(activations), self.kv_norm(activations)

    def forward(
        self,
        encoder_activations,
        backbone_activations,
        byte_position_ids,
        word_position_ids,
        cumulative_seq_lengths_per_word,
        position_embeddings,
        past_key_values,
        use_cache,
    ):
        if self.add_cross_attention:
            kv_activations = self.kv_norm(backbone_activations)
            q_activations = self.query_norm(encoder_activations)

            activations = self.cross_attention.forward(
                q_activations=q_activations,
                kv_activations=kv_activations,
                position_ids_q=byte_position_ids,
                position_ids_kv=word_position_ids,
                cumulative_seq_q=cumulative_seq_lengths_per_word,
                cumulative_seq_kv=torch.arange(0, kv_activations.size(1) + 1, device=encoder_activations.device, dtype=torch.int32),
                causal=False,
            )
            encoder_activations = encoder_activations + activations

        return self.llama_layer.forward(
            hidden_states=encoder_activations,
            position_ids=byte_position_ids,
            position_embeddings=position_embeddings,
            past_key_value=past_key_values,
            use_cache=use_cache,
        )[0]


class HATDecoder(nn.Module):
    def __init__(self, config: DecoderHATModelConfig, *args, **kwargs):
        super().__init__()

        self.decoder_layers = nn.Sequential()
        for layer_idx in range(config.num_hidden_layers):
            add_cross_attention = config.cross_attn_every_layer or layer_idx == 0
            self.decoder_layers.add_module(
                str(layer_idx),
                HATDecoderBlock(
                    add_cross_attention,
                    config,
                    layer_idx,
                ),
            )

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        backbone_activations: torch.Tensor,
        activations: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor | None = None,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = False,
    ) -> BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if byte_position_ids is None:
            past_seen_bytes = past_key_values.get_seq_length() if past_key_values is not None else 0
            byte_position_ids = torch.arange(
                past_seen_bytes,
                past_seen_bytes + activations.size(1),
                device=activations.device,
                dtype=torch.int32,
            ).unsqueeze(0)

        if cumulative_seq_lengths_per_word is None:
            cumulative_seq_lengths_per_word = torch.tensor([0, byte_position_ids.size(1)], dtype=byte_position_ids.dtype, device=byte_position_ids.device)

        if word_position_ids is None:
            raise ValueError()  # TODO

        position_embeddings = self.rotary_emb(activations, byte_position_ids)

        for _, layer in enumerate(self.decoder_layers):
            activations = layer(
                encoder_activations=activations,
                backbone_activations=backbone_activations,
                position_embeddings=position_embeddings,
                cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
                byte_position_ids=byte_position_ids,
                word_position_ids=word_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        return BaseModelOutputWithPast(
            last_hidden_state=activations,
            past_key_values=past_key_values if use_cache else None,
        )


class HATCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_size_q: int,
        hidden_size_kv: int,
        config: EncoderHATModelConfig | DecoderHATModelConfig,
        cross_attention_config: CrossAttentionConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_q = hidden_size_q
        self.hidden_size_kv = hidden_size_kv
        self.num_heads = cross_attention_config.num_attention_heads
        self.num_key_value_heads = cross_attention_config.attention_num_kv_heads
        self.num_repeat_kv = cross_attention_config.num_attention_heads // cross_attention_config.attention_num_kv_heads
        self.head_dim = hidden_size // self.num_heads
        self.key_query_norm = cross_attention_config.key_query_norm
        self.key_query_norm_per_head = cross_attention_config.key_query_norm_per_head

        self.q_proj = nn.Linear(
            in_features=hidden_size_q,
            out_features=hidden_size,
            dtype=dtype,
            bias=False,
        )

        self.k_proj = nn.Linear(
            in_features=hidden_size_kv,
            out_features=hidden_size // self.num_repeat_kv,
            dtype=dtype,
            bias=False,
        )

        self.v_proj = nn.Linear(
            in_features=hidden_size_kv,
            out_features=hidden_size // self.num_repeat_kv,
            dtype=dtype,
            bias=False,
        )

        if self.key_query_norm:
            if self.key_query_norm_per_head:
                # Both query and key have head dim equal to self.hidden_size_per_attention_head
                query_norm_dimensions = self.head_dim
                key_norm_dimensions = self.head_dim
            else:
                # Query dimensions across head is equal to hidden_size but key dimensions are divided
                # by self.num_repeat_kv
                query_norm_dimensions = self.hidden_size
                key_norm_dimensions = self.hidden_size // self.num_repeat_kv

            self.norm_query = RMSNorm(
                dimensions=query_norm_dimensions,
                eps=config.rms_norm_eps,
                device=self.q_proj.weight.device,
                dtype=dtype,
            )
            self.norm_key = RMSNorm(
                dimensions=key_norm_dimensions,
                eps=config.rms_norm_eps,
                device=self.q_proj.weight.device,
                dtype=dtype,
            )

        self.o_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size_q, dtype=dtype, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        q_activations: torch.Tensor,
        kv_activations: torch.Tensor,
        position_ids_q: torch.Tensor,
        position_ids_kv: torch.Tensor,
        cumulative_seq_kv: torch.Tensor,
        cumulative_seq_q: torch.Tensor,
        causal: bool = True,
        use_cache: bool = False,
        past_key_value: DynamicCache | None = None,
    ):
        q_len = cumulative_seq_q[-1]

        bsz, _, _ = kv_activations.size()
        query_states = self.q_proj(q_activations)
        key_states = self.k_proj(kv_activations)
        value_states = self.v_proj(kv_activations)

        if self.key_query_norm:
            assert self.norm_query is not None
            assert self.norm_key is not None
            # query_states and key_states are bsz seq_len (h d)
            if self.key_query_norm_per_head:
                # for per head qk norm we need head dim to be the last dim
                query_states = rearrange(
                    query_states,
                    "bsz seq_len (h d) -> bsz seq_len h d",
                    h=self.num_heads,
                )
                key_states = rearrange(
                    key_states,
                    "bsz seq_len (h d) -> bsz seq_len h d",
                    h=self.num_key_value_heads,
                )
            query_states = self.norm_query(query_states)
            key_states = self.norm_key(key_states)
            if self.key_query_norm_per_head:
                query_states = rearrange(
                    query_states,
                    "bsz seq_len h d -> bsz seq_len (h d)",
                )
                key_states = rearrange(
                    key_states,
                    "bsz seq_len h d -> bsz seq_len (h d)",
                )

        # TODO get rid of the double rearrange, this is just for compatibility with scaling
        query_states = rearrange(query_states, "bsz seq_len (h d) -> bsz h seq_len d", h=self.num_heads)
        key_states = rearrange(
            key_states,
            "bsz seq_len (h d) -> bsz h seq_len d",
            h=self.num_key_value_heads,
        )
        value_states = rearrange(
            value_states,
            "bsz seq_len (h d) -> bsz h seq_len d",
            h=self.num_key_value_heads,
        )

        # WIP: Should word_positions_id respect document boundaries?
        q_cos, q_sin = self.rotary_emb(query_states, position_ids_q)
        k_cos, k_sin = self.rotary_emb(key_states, position_ids_kv)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, q_cos=q_cos, q_sin=q_sin, k_cos=k_cos, k_sin=k_sin)

        query_states = rearrange(query_states, "bsz h seq_len d -> (bsz seq_len) h d")
        key_states = rearrange(key_states, "bsz h seq_len d -> (bsz seq_len) h d")
        value_states = rearrange(value_states, "bsz h seq_len d -> (bsz seq_len) h d")

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cumulative_seq_q,
            cu_seqlens_k=cumulative_seq_kv,
            max_seqlen_q=self._get_max_seqlen(cumulative_seq_q),
            max_seqlen_k=self._get_max_seqlen(cumulative_seq_kv),
            causal=False,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output

    def _get_max_seqlen(self, cumulative_word_lengths: torch.Tensor):
        diffs = cumulative_word_lengths[1:] - cumulative_word_lengths[:-1]
        return int(diffs.max().item())


class HATEncoderConnector(nn.Module):
    def __init__(
        self,
        config: EncoderHATModelConfig,
        backbone_hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_query = torch.nn.Parameter(
            torch.empty(
                1,
                1,
                backbone_hidden_size,
                device="cuda",
                dtype=dtype,
            )
        )

        self.cross_attention_encoder_connector = HATCrossAttention(
            hidden_size=config.cross_attention_config.hidden_size,
            hidden_size_q=backbone_hidden_size,
            hidden_size_kv=config.hidden_size,
            config=config,
            cross_attention_config=config.cross_attention_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor,
        word_position_ids: torch.Tensor,
        byte_position_ids: torch.Tensor,
    ):
        q_len = cumulative_seq_lengths_per_word.shape[0] - 1
        latent_query_repeated = self.latent_query.expand(-1, q_len, -1)
        cumulative_seq_lengths_q = torch.arange(
            start=0,
            end=latent_query_repeated.shape[1] + 1,
            step=1,
            device=self.latent_query.device,
            dtype=torch.int32,
        )
        word_embeddings = self.cross_attention_encoder_connector.forward(
            q_activations=latent_query_repeated,
            kv_activations=hidden_states,
            position_ids_q=word_position_ids,
            position_ids_kv=byte_position_ids,
            cumulative_seq_q=cumulative_seq_lengths_q,
            cumulative_seq_kv=cumulative_seq_lengths_per_word,
        )
        return word_embeddings


class HATEncoder(nn.Module):
    def __init__(
        self,
        config: EncoderHATModelConfig,
        dtype: torch.dtype = torch.bfloat16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        for layer in self.layers:
            layer.self_attn.sliding_window = config.sliding_window

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.word_window_size = config.cross_attention_config.word_window_size

    def forward(
        self,
        input_ids: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor | None = None,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,  # TODO: Remove
        past_key_values: DynamicCache | None = None,
        use_cache: bool | None = False,
    ):
        input_embeds = self.embedding_layer(input_ids)

        if cumulative_seq_lengths_per_word is None:
            cumulative_seq_lengths_per_word = torch.tensor([0, input_embeds.shape[1]], dtype=torch.int32, device=input_ids.device)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if byte_position_ids is None:
            past_seen_bytes = past_key_values.get_seq_length() if past_key_values is not None else 0
            byte_position_ids = torch.arange(
                past_seen_bytes,
                past_seen_bytes + input_embeds.shape[1],
                device=input_embeds.device,
            ).unsqueeze(0)

        if word_position_ids is None:
            raise ValueError()  # TODO

        hidden_states = input_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, byte_position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                position_ids=byte_position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return CausalLMOutputWithPast(
            hidden_states=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class HATForCausalLM(PreTrainedModel):
    config_class = HATArchitectureConfig
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def __init__(self, config: HATArchitectureConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.eos_token_id = config.eos_token_id
        self.encoder = HATEncoder(config.encoder_config)
        self.encoder_connector = HATEncoderConnector(config.encoder_config, config.backbone_config.hidden_size)
        self.backbone = HATBackbone(config.backbone_config)
        self.decoder_connector = HATDecoderConnector(config.backbone_config.hidden_size)
        self.decoder = HATDecoder(config.decoder_config)
        self.splitter = HATSplitter(special_token_dict=config.special_token_dict, max_word_size=config.max_word_size)
        self.layer_norm = RMSNorm(config.decoder_config.hidden_size, eps=config.decoder_config.rms_norm_eps, device=torch.device("cuda"), dtype=torch.bfloat16, norm_in_fp32=False)
        self.lm_head = nn.Linear(
            in_features=config.decoder_config.hidden_size,
            out_features=config.decoder_config.vocab_size,
            dtype=torch.bfloat16,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        byte_position_ids: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        past_key_values: HATCache | None = None,
        use_cache: bool = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is None and use_cache:
            past_key_values = HATCache()

        encoder_past_key_values = past_key_values.get_encoder_cache() if past_key_values is not None else None
        backbone_past_key_values = past_key_values.get_backbone_cache() if past_key_values is not None else None
        decoder_past_key_values = past_key_values.get_decoder_cache() if past_key_values is not None else None

        encoder_output: BaseModelOutputWithPast = self.encoder.forward(
            input_ids=input_ids,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
            past_key_values=encoder_past_key_values,
            use_cache=use_cache,
        )
        byte_level_activations = encoder_output.hidden_states

        encoder_connector_output = self.encoder_connector.forward(
            byte_level_activations,
            cumulative_seq_lengths_per_word,
            word_position_ids,
            byte_position_ids,
        )
        backbone_output: CausalLMOutputWithPast = self.backbone.forward(
            hidden_states=encoder_connector_output,
            position_ids=word_position_ids,
            past_key_values=backbone_past_key_values,
            use_cache=use_cache,
        )

        predictive_word_embeddings = self.decoder_connector.forward(backbone_activations=backbone_output.hidden_states)

        decoder_output = self.decoder.forward(
            activations=byte_level_activations,
            backbone_activations=predictive_word_embeddings,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
            past_key_values=decoder_past_key_values,
            use_cache=use_cache,
        )

        decoder_output = self.layer_norm(decoder_output.last_hidden_state)
        logits = self.lm_head(decoder_output)

        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=backbone_output.hidden_states,
            attentions=None,
        )

    def _append_byte(self, words: list[list[int]], token: int) -> list[list[int]]:
        extended_last_word = words.pop() + [token]
        try:
            text = self.splitter.decode(extended_last_word, errors='strict', skip_special_tokens=False)
            list_of_bytes = self.splitter.encode(text)
            words.extend([list(word_in_bytes) for word_in_bytes in list_of_bytes])
        except UnicodeDecodeError:
            # if decoding fails, the token cannot be part of a new word since it is not a valid
            # utf-8 end byte and we append it to the current word
            words.append(extended_last_word)
        return words

    def _split_encoder_activations(
        self,
        byte_encoder_activations: torch.Tensor,
        words: list[list[int]],
        previous_encoder_activations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split encoder activations between first word and next word.

        Args:
            byte_encoder_activations: Tensor of shape [batch_size, seq_len, hidden_size] containing all encoder activations which were computed in the current iteration
            words: List of word byte sequences which were completed in previous iteration and current iteration
            previous_encoder_activations: Optional tensor of shape [batch_size, prev_seq_len, hidden_size] containing precomputed activations from the previous iteration

        Returns:
            tuple containing:
                - first_word_encoder_activations: Tensor of shape [batch_size, first_word_len, hidden_size]
                - next_word_encoder_activations: Tensor of shape [batch_size, remaining_len, hidden_size]
        """

        assert sum(len(word) for word in words) - 1 == byte_encoder_activations.shape[1] + (previous_encoder_activations.shape[1] if previous_encoder_activations is not None else 0), "Length of (words - 1) must match the sum of byte_encoder_activations and previous_encoder_activations dimensions"

        next_word_encoder_activations = None
        if previous_encoder_activations is not None:
            # We have already precomputed first word's encoder activations partially in the previous iteration
            new_bytes_of_first_words = len(words[0]) - previous_encoder_activations.shape[1]
            # Concatenate the precomputed activations with the new activations that still belong to the first word
            first_word_encoder_activations = torch.cat([previous_encoder_activations, byte_encoder_activations[:, :new_bytes_of_first_words]], dim=1)
            if len(words[1]) > 1:
                # The remaining activations that belong to the next word
                next_word_encoder_activations = byte_encoder_activations[:, new_bytes_of_first_words:]
            else:
                next_word_encoder_activations = None
        else:
            # We have not precomputed any activations for the first word previously
            first_word_encoder_activations = byte_encoder_activations[:, : len(words[0])]

            if len(words[1]) > 1:
                next_word_encoder_activations = byte_encoder_activations[:, len(words[0]) :]
            else:
                next_word_encoder_activations = None

        return first_word_encoder_activations, next_word_encoder_activations

    def _complete_word(
        self,
        input_ids: torch.Tensor,
        byte_position_ids: torch.Tensor,
        predictive_word_embeddings: torch.Tensor,
        word_position_id: torch.Tensor,
        encoder_cache: DynamicCache,
        decoder_cache: DynamicCache,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        previous_encoder_activations: torch.Tensor | None = None,
    ):
        """Generate byte tokens until we hit the first byte of a new word."""
        words: list[list[int]] = [input_ids.squeeze(0).tolist()]
        byte_encoder_activations: list[torch.Tensor] = []
        completion_logits: list[torch.Tensor] = []

        if previous_encoder_activations is not None:
            # we need to pass all inputs in order to get the correct encoding/decoding by the splitter
            # but only the last byte is used for the generation
            # since the cache is already populated with the first word's activations
            input_ids = input_ids[:, -1:]

        while True:
            encoder_output = self.encoder.forward(
                input_ids,
                byte_position_ids=None,
                word_position_ids=word_position_id,
                past_key_values=encoder_cache,
                use_cache=True,
            )
            byte_encoder_activations.append(encoder_output.hidden_states)
            decoder_output = self.decoder.forward(
                predictive_word_embeddings,
                encoder_output.hidden_states,
                byte_position_ids=None,
                word_position_ids=word_position_id,
                past_key_values=decoder_cache,
                use_cache=True,
            )
            decoder_output = self.layer_norm(decoder_output.last_hidden_state)
            logits = self.lm_head(decoder_output)
            completion_logits.append(logits[0, -1:, :])
            next_byte = int(sample_fn(logits).item())
            words = self._append_byte(words, next_byte)
            if len(words) > 1 or next_byte == self.eos_token_id:
                byte_encoder_activations = torch.cat(byte_encoder_activations, dim=1)
                first_word_encoder_activations, next_word_encoder_activations = self._split_encoder_activations(
                    byte_encoder_activations,
                    words,
                    previous_encoder_activations,
                )
                break
            input_ids = torch.tensor([[next_byte]], dtype=input_ids.dtype, device=input_ids.device)

        num_kv = encoder_cache.get_seq_length()

        completion = sum(words, [])[-len(completion_logits) :]
        if next_word_encoder_activations is not None:
            start_idx = num_kv - first_word_encoder_activations.shape[1] - next_word_encoder_activations.shape[1]
            end_idx = num_kv - next_word_encoder_activations.shape[1]
            # We do not want to return the logits for the second word went into the mulitbyte starting character case
            # When that happens we remove the logits and post-hoc fix the decoder cache and compute new logits
            # This is breaking causality but we want to imitate uncached generation/training behavior
            completion_logits = completion_logits[:-next_word_encoder_activations.shape[1]]
        else:
            start_idx = num_kv - first_word_encoder_activations.shape[1]
            end_idx = num_kv

        byte_position_ids = torch.arange(start_idx, end_idx, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        completed_word_embedding = self.encoder_connector.forward(
            first_word_encoder_activations,
            cumulative_seq_lengths_per_word=torch.tensor([0, first_word_encoder_activations.size(1)], dtype=torch.int32, device=input_ids.device),
            word_position_ids=word_position_id,
            byte_position_ids=byte_position_ids,
        )

        bytes_of_next_word = words[1]

        return (
            completion,
            completed_word_embedding,
            bytes_of_next_word,
            byte_position_ids[:, -1].item() + 1,
            completion_logits,
            next_word_encoder_activations,
        )

    def _populate_cache(
        self,
        input_ids: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor,
        word_position_ids: torch.Tensor,
    ):
        last_word_start = cumulative_seq_lengths_per_word[-2]
        last_word_end = cumulative_seq_lengths_per_word[-1]

        # Populate cache with everything except last word
        initial_forward_output = self.forward(
            input_ids=input_ids[:, :last_word_start],
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word[:-1],
            byte_position_ids=byte_position_ids[:, :last_word_start],
            word_position_ids=word_position_ids[:, :-1],
            past_key_values=None,
            use_cache=True,
        )
        return initial_forward_output, last_word_start, last_word_end

    def _initialize_generation_state(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
    ):
        max_total_bytes = max_new_tokens + input_ids.shape[1]
        if byte_position_ids is None:
            byte_position_ids = torch.arange(0, cumulative_seq_lengths_per_word[-1].item(), device=input_ids.device, dtype=torch.int32).unsqueeze(0)

        if word_position_ids is None:
            word_position_ids = torch.arange(0, cumulative_seq_lengths_per_word.shape[0] - 1, device=input_ids.device, dtype=torch.int32).unsqueeze(0)

        initial_forward_output, last_word_start, last_word_end = self._populate_cache(
            input_ids=input_ids,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
        )

        completion_bytes: list[int] = []
        completion_logits: list[torch.Tensor] = []
        # Slice input_ids and byte_position_ids to only contain the last word for the generation loop
        current_input_ids = input_ids[:, last_word_start:last_word_end]
        next_byte_id = last_word_end.item()  # Ensure this is an int
        current_byte_position_ids = byte_position_ids[:, last_word_start:last_word_end]
        current_word_position_id = word_position_ids[:, -1].unsqueeze(-1)
        backbone_last_hidden_state = initial_forward_output.hidden_states[:, -1:, :]
        next_word_encoder_activations = None
        return (
            initial_forward_output,
            completion_bytes,
            completion_logits,
            current_input_ids,
            next_byte_id,
            current_byte_position_ids,
            current_word_position_id,
            backbone_last_hidden_state,
            next_word_encoder_activations,
            max_total_bytes,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        use_cache: bool = True,
        stop_sequences: Sequence[str] | None = None,
    ):
        if use_cache:
            completion_text, completion_logits = self._generate_cached(input_ids, max_new_tokens, cumulative_seq_lengths_per_word, byte_position_ids, word_position_ids, sample_fn, stop_sequences=stop_sequences)
        else:
            completion_text, completion_logits = self._generate_uncached(input_ids, max_new_tokens, cumulative_seq_lengths_per_word, byte_position_ids, word_position_ids, sample_fn, stop_sequences=stop_sequences)

        # remove stop sequence if exists
        if stop_sequences is not None:
            stop_sequences = sorted(stop_sequences, key=lambda i: len(i), reverse=True)
            for stop_sequence in stop_sequences:
                if stop_sequence in completion_text:
                    completion_text_left = completion_text.split(stop_sequence)[0]
                    completion_text_removed = completion_text[len(completion_text_left) :]

                    completion_logits = completion_logits[: -len(list(bytes(completion_text_removed.encode("UTF-8"))))]
                    completion_text = completion_text_left
                    break

        return ModelOutput(
            completion_text=completion_text,
            input_ids=input_ids,
            completion_logits=completion_logits,
        )

    def _fix_decoder_cache(self, predictive_word_embeddings: torch.Tensor, encoder_activions: torch.Tensor, decoder_cache: DynamicCache, word_position_id: torch.Tensor):
        decoder_cache.crop(decoder_cache.get_seq_length() - encoder_activions.shape[1])
        real_decoder_logits = self.decoder.forward(
            predictive_word_embeddings,
            encoder_activions,
            byte_position_ids=None,
            word_position_ids=word_position_id,
            past_key_values=decoder_cache,
        ).last_hidden_state

        decoder_output = self.layer_norm(real_decoder_logits)
        logits = self.lm_head(decoder_output)
        return logits

    @torch.no_grad()
    def _generate_cached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        stop_sequences: Sequence[str] | None = None,
    ):
        (
            initial_forward_output,
            completion_bytes,  # empty list
            completion_logits,  # empty list
            input_ids,  # This is now the sliced input_ids for the last word
            next_byte_id,
            byte_position_ids,  # This is now the sliced byte_position_ids for the last word
            word_position_id,
            backbone_last_hidden_state,
            next_word_encoder_activations,  # None for the first iteration
            max_total_bytes,
        ) = self._initialize_generation_state(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
        )

        while next_byte_id < max_total_bytes:
            completion, completed_word_embedding, bytes_of_next_word, next_byte_id, next_completion_logits, next_word_encoder_activations = self._complete_word(
                input_ids=input_ids,
                byte_position_ids=byte_position_ids,
                predictive_word_embeddings=backbone_last_hidden_state,
                word_position_id=word_position_id,
                encoder_cache=initial_forward_output.past_key_values.get_encoder_cache(),
                decoder_cache=initial_forward_output.past_key_values.get_decoder_cache(),
                sample_fn=sample_fn,
                previous_encoder_activations=next_word_encoder_activations,
            )
            completion_logits.extend(next_completion_logits)
            completion_bytes.extend(completion)

            if self.eos_token_id in completion_bytes:
                completion_bytes = completion_bytes[: completion_bytes.index(self.eos_token_id)]
                break

            if stop_sequences is not None:
                try:
                    completion_text_tmp = self.splitter.decode(completion_bytes)
                    if any(stop_sequence in completion_text_tmp for stop_sequence in stop_sequences):
                        break
                except Exception as e:
                    print("Cannot compare stop sequence", e)

            backbone_output = self.backbone.forward(
                hidden_states=completed_word_embedding,
                position_ids=None,
                past_key_values=initial_forward_output.past_key_values.get_backbone_cache(),
                use_cache=True,
            )
            backbone_last_hidden_state = backbone_output.hidden_states[:, -1, :].unsqueeze(1)

            word_position_id = word_position_id + 1
            if len(bytes_of_next_word) > 1:
                real_decoder_logits = self._fix_decoder_cache(
                    predictive_word_embeddings=backbone_last_hidden_state,
                    encoder_activions=next_word_encoder_activations,
                    decoder_cache=initial_forward_output.past_key_values.get_decoder_cache(),
                    word_position_id=word_position_id,
                )
                completion_logits.extend(real_decoder_logits)

            input_ids = torch.tensor([bytes_of_next_word], dtype=input_ids.dtype, device=input_ids.device)
            byte_position_ids = torch.tensor([[next_byte_id]], dtype=input_ids.dtype, device=input_ids.device)

        completion_bytes = completion_bytes[:max_new_tokens]
        completion_logits = torch.cat(completion_logits[:max_new_tokens], dim=0)
        completion_text = self.splitter.decode(completion_bytes)

        return completion_text, completion_logits

    @torch.no_grad()
    def _generate_uncached(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        stop_sequences: Sequence[str] | None = None,
    ):
        if byte_position_ids is None:
            byte_position_ids = torch.arange(0, cumulative_seq_lengths_per_word[-1].item(), device=input_ids.device, dtype=torch.int32).unsqueeze(0)

        if word_position_ids is None:
            word_position_ids = torch.arange(0, cumulative_seq_lengths_per_word.shape[0] - 1, device=input_ids.device, dtype=torch.int32).unsqueeze(0)

        word_list = []
        for i in range(1, cumulative_seq_lengths_per_word.shape[0]):
            start_idx = cumulative_seq_lengths_per_word[i - 1]
            end_idx = cumulative_seq_lengths_per_word[i]
            word_list.append(input_ids[:, start_idx:end_idx].squeeze(0).tolist())

        completion_bytes = []
        for _ in range(max_new_tokens):
            output = self.forward(
                input_ids=input_ids,
                cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
                byte_position_ids=byte_position_ids,
                word_position_ids=word_position_ids,
                past_key_values=None,
            )

            next_byte = int(sample_fn(output.logits).item())
            completion_bytes.append(next_byte)
            if next_byte == self.eos_token_id:
                break
            word_list = self._append_byte(word_list, next_byte)

            input_ids = torch.tensor(sum(word_list, []), dtype=torch.long, device=input_ids.device).unsqueeze(0)
            cumulative_seq_lengths_per_word = torch.tensor([0] + list(itertools.accumulate(len(word) for word in word_list if len(word) > 0)), dtype=torch.int32, device=input_ids.device)
            byte_position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device, dtype=torch.int32).unsqueeze(0)
            word_position_ids = torch.arange(0, cumulative_seq_lengths_per_word.shape[0] - 1, device=input_ids.device, dtype=torch.int32).unsqueeze(0)

            if stop_sequences is not None:
                try:
                    completion_text_tmp = self.splitter.decode(completion_bytes)
                    if any(completion_text_tmp.endswith(stop_sequence) for stop_sequence in stop_sequences):
                        break
                except Exception as e:
                    print("Cannot compare stop sequence", e)

        completion_text = self.splitter.decode(completion_bytes)
        completion_logits = output.logits[0, -len(completion_bytes) :, :]

        return completion_text, completion_logits

    def _prepare_input(self, input_str: str, add_llama_template: bool = True, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if add_llama_template:
            input_str = LLAMA_TEMPLATE.format(input=input_str)

        if device is None:
            assert torch.cuda.is_available(), "CUDA is not available"
            device = torch.device("cuda")
        input_ids_list = []
        cumulative_per_word_lengths_list = [0]

        words = self.splitter.encode(input_str)
        for word in words:
            input_ids_list.extend(word)
            word_length = len(word)
            cumulative_per_word_lengths_list.append(cumulative_per_word_lengths_list[-1] + word_length)
        input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.int32).unsqueeze(0)
        cumulative_per_word_lengths = torch.tensor(cumulative_per_word_lengths_list, device=device, dtype=torch.int32)
        return input_ids, cumulative_per_word_lengths
