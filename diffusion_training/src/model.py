import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from hat.model import HATDecoder
from src.utils import ModelConfig


def _word_positions(boundaries: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.arange(boundaries.numel() - 1, device=device, dtype=torch.int32).unsqueeze(0)


def _byte_positions(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=torch.int32).unsqueeze(0)


def build_compression_mlp(
    in_dim: int, out_dim: int, num_layers: int, hidden_dims: list[int] | None = None
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError(f"latent_compression_layers must be >= 1, got {num_layers}.")
    if hidden_dims is not None:
        if len(hidden_dims) != num_layers - 1:
            raise ValueError(
                f"latent_compression_hidden_dims has {len(hidden_dims)} widths but "
                f"latent_compression_layers={num_layers} needs exactly {num_layers - 1}."
            )
        dims = [in_dim, *(int(width) for width in hidden_dims), out_dim]
    else:
        ratio = out_dim / in_dim
        dims = [round(in_dim * ratio ** (i / num_layers)) for i in range(num_layers + 1)]
        dims[0], dims[-1] = in_dim, out_dim  # pin endpoints exactly
    layers: list[nn.Module] = []
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def build_decoder_config(base_config: Any, model_config: ModelConfig):
    decoder_config = copy.deepcopy(base_config.decoder_config)
    decoder_config.num_hidden_layers = model_config.decoder_layers
    decoder_config.hidden_size = model_config.decoder_hidden_size
    decoder_config.num_attention_heads = model_config.decoder_self_attention_heads
    decoder_config.num_key_value_heads = model_config.decoder_self_attention_heads
    decoder_config.vocab_size = 256
    decoder_config.cross_attn_every_layer = True

    cross_config = decoder_config.cross_attention_config
    cross_config.hidden_size_q = model_config.decoder_hidden_size
    cross_config.hidden_size_kv = base_config.backbone_config.hidden_size
    cross_config.num_attention_heads = model_config.decoder_cross_attention_heads
    cross_config.attention_num_kv_heads = model_config.decoder_cross_attention_heads
    cross_config.hidden_size = model_config.decoder_cross_attention_heads * model_config.head_size
    return decoder_config


class LatentHATDecoder(HATDecoder):
    """HAT decoder variant that embeds teacher-forced bytes itself."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.byte_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        byte_ids: torch.Tensor,
        z_words: torch.Tensor,
        cumulative_seq_lengths_per_word: torch.Tensor,
        byte_position_ids: torch.Tensor | None = None,
        word_position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
    ):
        activations = self.byte_embedding(byte_ids.long())
        return super().forward(
            backbone_activations=z_words,
            activations=activations,
            cumulative_seq_lengths_per_word=cumulative_seq_lengths_per_word,
            byte_position_ids=byte_position_ids,
            word_position_ids=word_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )


class DiffusionAutoencoder(nn.Module):
    """Frozen HAT encoder/connector plus a fresh latent-conditioned byte decoder."""

    def __init__(
        self,
        pretrained_hat: nn.Module,
        model_config: ModelConfig,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.config = model_config
        self.encoder = pretrained_hat.encoder
        self.encoder_connector = pretrained_hat.encoder_connector
        self.splitter = getattr(pretrained_hat, "splitter", None)

        decoder_config = build_decoder_config(pretrained_hat.config, model_config)
        self.decoder = LatentHATDecoder(decoder_config)
        self.decoder_norm = nn.RMSNorm(decoder_config.hidden_size, eps=decoder_config.rms_norm_eps)
        self.lm_head = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=False)

        backbone_hidden = pretrained_hat.config.backbone_config.hidden_size
        if model_config.latent_compression_dim is not None:
            dim = model_config.latent_compression_dim
            layers = model_config.latent_compression_layers
            hidden_dims = model_config.latent_compression_hidden_dims
            self.encoder_compression = build_compression_mlp(
                backbone_hidden, dim, layers, hidden_dims=hidden_dims
            )
            self.decoder_decompression = build_compression_mlp(
                dim,
                backbone_hidden,
                layers,
                hidden_dims=list(reversed(hidden_dims)) if hidden_dims else None,
            )
        else:
            self.encoder_compression = None
            self.decoder_decompression = None

        if freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def from_pretrained(
        cls,
        model_config: ModelConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        device_map: str | dict[str, int] | None = None,
        freeze_encoder: bool = True,
        local_files_only: bool = False,
    ) -> "DiffusionAutoencoder":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("The vendored HAT modules allocate CUDA tensors; use TinyDiffusionAutoencoder for CPU smoke tests.")

        pretrained = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.pretrained_model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=local_files_only,
        )
        if device_map is None:
            pretrained = pretrained.to(device=device, dtype=torch_dtype)
        model = cls(pretrained, model_config=model_config, freeze_encoder=freeze_encoder)
        if device_map is None:
            model = model.to(device=device, dtype=torch_dtype)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
            self.encoder_connector.eval()
        return self

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder_connector.requires_grad_(False)
        self.encoder.eval()
        self.encoder_connector.eval()

    def _boundary_list(
        self,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if isinstance(word_boundaries, torch.Tensor):
            if word_boundaries.ndim == 1:
                return [word_boundaries.to(device=device, dtype=torch.int32)]
            return [word_boundaries[index].to(device=device, dtype=torch.int32) for index in range(batch_size)]
        return [boundary.to(device=device, dtype=torch.int32) for boundary in word_boundaries]

    def encode(
        self,
        byte_ids: torch.Tensor,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor]:
        device = byte_ids.device
        boundaries = self._boundary_list(word_boundaries, byte_ids.size(0), device)
        z_words = []
        for index, boundary in enumerate(boundaries):
            length = int(boundary[-1].item())
            sample_ids = byte_ids[index : index + 1, :length].to(dtype=torch.long)
            byte_position_ids = _byte_positions(length, device)
            word_position_ids = _word_positions(boundary, device)
            encoder_output = self.encoder(
                input_ids=sample_ids,
                cumulative_seq_lengths_per_word=boundary,
                byte_position_ids=byte_position_ids,
                word_position_ids=word_position_ids,
                use_cache=False,
            )
            z = self.encoder_connector(
                encoder_output.hidden_states,
                boundary,
                word_position_ids,
                byte_position_ids,
            )
            if self.encoder_compression is not None:
                z = self.encoder_compression(z)
            z_words.append(z)
        return z_words

    def decode(
        self,
        byte_ids: torch.Tensor,
        z_words: Sequence[torch.Tensor] | torch.Tensor,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        device = byte_ids.device
        boundaries = self._boundary_list(word_boundaries, byte_ids.size(0), device)
        if isinstance(z_words, torch.Tensor):
            z_list = [z_words]
        else:
            z_list = list(z_words)

        logits = byte_ids.new_zeros((*byte_ids.shape, 256), dtype=torch.float32)
        for index, boundary in enumerate(boundaries):
            length = int(boundary[-1].item())
            sample_ids = byte_ids[index : index + 1, :length]
            byte_position_ids = _byte_positions(length, device)
            word_position_ids = _word_positions(boundary, device)
            z = z_list[index]
            if self.decoder_decompression is not None:
                z = self.decoder_decompression(z)
            decoder_output = self.decoder(
                byte_ids=sample_ids,
                z_words=z,
                cumulative_seq_lengths_per_word=boundary,
                byte_position_ids=byte_position_ids,
                word_position_ids=word_position_ids,
                use_cache=False,
            )
            hidden = self.decoder_norm(decoder_output.last_hidden_state)
            logits[index : index + 1, :length, :] = self.lm_head(hidden).float()
        return logits

    def forward(
        self,
        byte_ids: torch.Tensor | None = None,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **batch: Any,
    ) -> dict[str, torch.Tensor]:
        if byte_ids is None:
            byte_ids = batch.get("byte_ids", batch.get("input_ids"))
        if word_boundaries is None:
            word_boundaries = batch.get("word_boundaries", batch.get("cumulative_seq_lengths_per_word"))
        if labels is None:
            labels = batch.get("labels")
        if byte_ids is None or word_boundaries is None:
            raise ValueError("byte_ids and word_boundaries are required.")

        z_words = self.encode(byte_ids, word_boundaries)
        logits = self.decode(byte_ids, z_words, word_boundaries)
        output = {"logits": logits, "z_words": z_words}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return output

    def forward_from_latents(
        self,
        byte_ids: torch.Tensor | None = None,
        z_words: list[torch.Tensor] | None = None,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        if byte_ids is None or z_words is None or word_boundaries is None:
            raise ValueError("byte_ids, z_words, and word_boundaries are required.")
        logits = self.decode(byte_ids, z_words, word_boundaries)
        output = {"logits": logits, "z_words": z_words}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return output


class TinyDiffusionAutoencoder(nn.Module):
    """CPU-friendly model with the same batch contract as DiffusionAutoencoder."""

    def __init__(self, hidden_size: int = 64, latent_size: int = 128, vocab_size: int = 256):
        super().__init__()
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder_proj = nn.Linear(hidden_size, latent_size)
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.mixer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def encode(
        self,
        byte_ids: torch.Tensor,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor,
    ) -> list[torch.Tensor]:
        if isinstance(word_boundaries, torch.Tensor):
            boundaries = [word_boundaries] if word_boundaries.ndim == 1 else list(word_boundaries)
        else:
            boundaries = list(word_boundaries)

        embedded = self.byte_embedding(byte_ids.long())
        z_words = []
        for batch_index, boundary in enumerate(boundaries):
            pieces = []
            for start, end in zip(boundary[:-1], boundary[1:]):
                start_i = int(start.item())
                end_i = int(end.item())
                pieces.append(embedded[batch_index, start_i:end_i].mean(dim=0))
            z_words.append(self.encoder_proj(torch.stack(pieces, dim=0)).unsqueeze(0))
        return z_words

    def decode(
        self,
        byte_ids: torch.Tensor,
        z_words: Sequence[torch.Tensor],
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(word_boundaries, torch.Tensor):
            boundaries = [word_boundaries] if word_boundaries.ndim == 1 else list(word_boundaries)
        else:
            boundaries = list(word_boundaries)

        hidden = self.decoder_embedding(byte_ids.long())
        for batch_index, boundary in enumerate(boundaries):
            for word_index, (start, end) in enumerate(zip(boundary[:-1], boundary[1:])):
                start_i = int(start.item())
                end_i = int(end.item())
                hidden[batch_index, start_i:end_i] = hidden[batch_index, start_i:end_i] + self.latent_to_hidden(
                    z_words[batch_index][0, word_index]
                )
        return self.lm_head(hidden + self.mixer(hidden))

    def forward(
        self,
        byte_ids: torch.Tensor | None = None,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **batch: Any,
    ) -> dict[str, torch.Tensor]:
        if byte_ids is None:
            byte_ids = batch.get("byte_ids", batch.get("input_ids"))
        if word_boundaries is None:
            word_boundaries = batch.get("word_boundaries", batch.get("cumulative_seq_lengths_per_word"))
        if labels is None:
            labels = batch.get("labels")
        if byte_ids is None or word_boundaries is None:
            raise ValueError("byte_ids and word_boundaries are required.")

        z_words = self.encode(byte_ids, word_boundaries)
        logits = self.decode(byte_ids, z_words, word_boundaries)
        output = {"logits": logits, "z_words": z_words}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return output

    def forward_from_latents(
        self,
        byte_ids: torch.Tensor | None = None,
        z_words: list[torch.Tensor] | None = None,
        word_boundaries: Sequence[torch.Tensor] | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if byte_ids is None or z_words is None or word_boundaries is None:
            raise ValueError("byte_ids, z_words, and word_boundaries are required.")
        logits = self.decode(byte_ids, z_words, word_boundaries)
        output = {"logits": logits, "z_words": z_words}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return output
