from dataclasses import dataclass

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig


@dataclass
class TransformerHATModelConfig(LlamaConfig):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        intermediate_size: int,
        max_position_embeddings: int,
        rope_scaling: dict,
        rope_theta: float,
        mlp_bias: bool,
        use_cache: bool = True,
        sliding_window: int | None = None,
        vocab_size: int = 0,
        hidden_act: str = "silu",
        key_query_norm: bool = False,
        key_query_norm_per_head: bool = False,
        is_neox_style: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            mlp_bias=mlp_bias,
            use_cache=use_cache,
            **kwargs,
        )

        self.sliding_window = sliding_window
        self.key_query_norm = key_query_norm
        self.key_query_norm_per_head = key_query_norm_per_head
        self.is_neox_style = is_neox_style

    def to_dict(self):
        config_dict = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "rms_norm_eps": self.rms_norm_eps,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_scaling": self.rope_scaling,
            "rope_theta": self.rope_theta,
            "mlp_bias": self.mlp_bias,
            "use_cache": self.use_cache,
            "sliding_window": self.sliding_window,
            "transformers_version": self.transformers_version,
            "key_query_norm": self.key_query_norm,
            "key_query_norm_per_head": self.key_query_norm_per_head,
            "is_neox_style": self.is_neox_style,
        }
        return config_dict


@dataclass
class CrossAttentionConfig:
    def __init__(
        self,
        hidden_size: int,
        hidden_size_q: int,
        hidden_size_kv: int,
        num_attention_heads: int,
        attention_num_kv_heads: int,
        word_window_size: int,
        key_query_norm: bool, 
        key_query_norm_per_head: bool,
    ):
        self.hidden_size = hidden_size
        self.hidden_size_q = hidden_size_q
        self.hidden_size_kv = hidden_size_kv
        self.num_attention_heads = num_attention_heads
        self.attention_num_kv_heads = attention_num_kv_heads
        self.word_window_size = word_window_size
        self.key_query_norm = key_query_norm
        self.key_query_norm_per_head = key_query_norm_per_head

    def to_dict(self):
        return {
            "hidden_size_q": self.hidden_size_q,
            "hidden_size_kv": self.hidden_size_kv,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "attention_num_kv_heads": self.attention_num_kv_heads,
            "word_window_size": self.word_window_size,
            "key_query_norm": self.key_query_norm,
            "key_query_norm_per_head": self.key_query_norm_per_head,
        }


@dataclass
class DecoderHATModelConfig(TransformerHATModelConfig):
    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        sliding_window: int,
        cross_attention_config: CrossAttentionConfig,
        cross_attn_every_layer: bool,
        **kwargs,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            **kwargs,
        )
        self.cross_attn_every_layer = cross_attn_every_layer
        self.cross_attention_config = cross_attention_config

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict["cross_attn_every_layer"] = self.cross_attn_every_layer
        config_dict["cross_attention_config"] = self.cross_attention_config.to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = config_dict.copy()  # Avoid modifying the original dict
        config_dict.update(kwargs)  # Apply overrides
        dict_config = config_dict.pop("cross_attention_config", {})
        cross_attention_config = CrossAttentionConfig(**dict_config)
        config_dict["cross_attention_config"] = cross_attention_config
        return cls(**config_dict)


@dataclass
class EncoderHATModelConfig(TransformerHATModelConfig):
    def __init__(
        self,
        cross_attention_config: CrossAttentionConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_config = cross_attention_config

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = config_dict.copy()  # Avoid modifying the original dict
        config_dict.update(kwargs)  # Apply overrides
        dict_config = config_dict.pop("cross_attention_config", {})
        cross_attention_config = CrossAttentionConfig(**dict_config)
        config_dict["cross_attention_config"] = cross_attention_config

        return cls(**config_dict)

    def to_dict(self):
        config_dict = super().to_dict()
        if self.cross_attention_config:
            config_dict["cross_attention_config"] = self.cross_attention_config.to_dict()
        return config_dict


@dataclass
class HATArchitectureConfig(PretrainedConfig):
    model_type: str = "hierarchical_autoregressive_transformer"

    def __init__(
        self,
        special_token_dict: dict | None = None,
        encoder_config: EncoderHATModelConfig | None = None,
        backbone_config: TransformerHATModelConfig | None = None,
        decoder_config: DecoderHATModelConfig | None = None,
        model_type: str = "hierarchical_autoregressive_transformer",
        eos_token_id: int = 192,
        max_word_size: int = 100,
        sliding_window: int = 768,
        max_position_embeddings: int = 262144,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.backbone_config = backbone_config
        self.decoder_config = decoder_config
        self.model_type = model_type
        self.eos_token_id = eos_token_id
        self.max_word_size = max_word_size
        self.special_token_dict = special_token_dict
        self.transformers_version = "4.46.3"

        # set these for out of the box vllm inference
        self.architectures = ["HATDecoderForCausalLM"]
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.torch_dtype = "bfloat16"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiates a HATArchitectureConfig from a Python dictionary of parameters.

        Overrides the base `from_dict` to correctly handle nested config objects.
        """
        config_dict = config_dict.copy()  # Avoid modifying the original dict
        config_dict.update(kwargs)  # Apply overrides

        # Pop and instantiate nested config dictionaries
        encoder_dict = config_dict.pop("encoder_config", {})
        backbone_dict = config_dict.pop("backbone_config", {})
        decoder_dict = config_dict.pop("decoder_config", {})

        # Instantiate nested configs
        encoder_config = EncoderHATModelConfig.from_dict(encoder_dict) if encoder_dict else None
        backbone_config = TransformerHATModelConfig.from_dict(backbone_dict) if backbone_dict else None
        decoder_config = DecoderHATModelConfig.from_dict(decoder_dict) if decoder_dict else None
        special_token_dict = config_dict.pop("special_token_dict", {"<|eot_id|>": 192})
        max_word_size = config_dict.pop("max_word_size", 100)
        return_unused_kwargs = config_dict.pop("return_unused_kwargs", False)
        if return_unused_kwargs:
            return cls(
                encoder_config=encoder_config,
                backbone_config=backbone_config,
                decoder_config=decoder_config,
                special_token_dict=special_token_dict,
                max_word_size=max_word_size,
                **config_dict,
            ), {}
        else:
            return cls(
                encoder_config=encoder_config,
                backbone_config=backbone_config,
                decoder_config=decoder_config,
                special_token_dict=special_token_dict,
                max_word_size=max_word_size,
                **config_dict,
            )

    def to_dict(self):
        config_dict = {}
        if self.encoder_config:
            config_dict["encoder_config"] = self.encoder_config.to_dict()
        if self.backbone_config:
            config_dict["backbone_config"] = self.backbone_config.to_dict()
        if self.decoder_config:
            config_dict["decoder_config"] = self.decoder_config.to_dict()
        config_dict["model_type"] = self.model_type
        config_dict["transformers_version"] = self.transformers_version
        config_dict["auto_map"] = {"AutoConfig": "config.HATArchitectureConfig", "AutoModelForCausalLM": "model.HATForCausalLM"}
        config_dict["special_token_dict"] = self.special_token_dict
        
        # print these out to the config for vllm
        config_dict["max_word_size"] = self.max_word_size
        config_dict["sliding_window"] = self.sliding_window
        config_dict["max_position_embeddings"] = self.max_position_embeddings
        config_dict["torch_dtype"] = self.torch_dtype
        config_dict["architectures"] = self.architectures
        return config_dict


class EncoderHATModel(nn.Module):
    def __init__(self, config: HATArchitectureConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
