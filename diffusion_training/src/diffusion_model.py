import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormZero

from src.utils import DiffusionModelConfig


def uses_context_channel(config: DiffusionModelConfig) -> bool:
    """True when the separate-channel prefix conditioning is active (plain_dit only)."""
    return getattr(config, "prefix_cond_enabled", False) and getattr(config, "prefix_cond_mode", "data") == "channel"


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """Create sinusoidal embeddings for continuous times in [0, 1]."""

    if t.ndim != 1:
        t = t.reshape(-1)
    half_dim = dim // 2
    if half_dim == 0:
        return t[:, None]

    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / max(1, half_dim - 1)
    )
    args = t.float()[:, None] * scale * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PlainTimestepEmbedder(nn.Module):
    """Timestep embedder from the original DiT implementation."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class PlainLabelEmbedder(nn.Module):
    """Single dummy class embedder, preserving the original DiT conditioning path."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(1, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class PlainPatchEmbed(nn.Module):
    """Minimal PatchEmbed equivalent for stock-DiT shaped latent images."""

    def __init__(self, input_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size.")
        self.input_size = input_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class PlainAttention(nn.Module):
    """ViT attention matching the shape contract used by the original DiT blocks."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, hidden_size = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, hidden_size)
        return self.proj_drop(self.proj(x))


class PlainMlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class PlainDiTBlock(nn.Module):
    """Original DiT-style block with explicit six-way adaLN-Zero modulation."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = PlainAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = PlainMlp(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PlainFinalLayer(nn.Module):
    """Original DiT final layer: adaLN modulation followed by patch projection."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("2D sin-cos positional embeddings require an even embed_dim.")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("1D sin-cos positional embeddings require an even embed_dim.")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class LatentFlowTransformer(nn.Module):
    """Bidirectional Transformer that predicts flow velocity for word latents."""

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        if config.model_dim % config.num_heads != 0:
            raise ValueError("diffusion.model_dim must be divisible by diffusion.num_heads.")
        if config.objective != "flow_matching":
            raise ValueError(f"Unsupported diffusion objective: {config.objective}")
        if config.path not in ("linear", "cosine"):
            raise ValueError(f"Unsupported diffusion path: {config.path}")
        if config.t_sampling != "uniform":
            raise ValueError(f"Unsupported t_sampling: {config.t_sampling}")
        if config.self_conditioning:
            raise ValueError("self_conditioning requires the dit backbone")

        self.config = config
        self.input_norm = nn.LayerNorm(config.latent_dim)
        self.input_proj = nn.Linear(config.latent_dim, config.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_words, config.model_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.model_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.latent_dim)

        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        z_self: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_t.ndim != 3:
            raise ValueError(f"Expected z_t with shape [batch, words, latent_dim], got {tuple(z_t.shape)}.")
        batch_size, word_count, latent_dim = z_t.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {latent_dim}.")
        if word_count > self.config.max_words:
            raise ValueError(f"word_count={word_count} exceeds max_words={self.config.max_words}.")

        if mask is None:
            mask = torch.ones((batch_size, word_count), device=z_t.device, dtype=torch.bool)
        else:
            mask = mask.to(device=z_t.device, dtype=torch.bool)

        h = self.input_proj(self.input_norm(z_t))
        h = h + self.position_embedding[:, :word_count, :].to(dtype=h.dtype)
        time_embedding = sinusoidal_time_embedding(t.to(z_t.device), self.config.model_dim)
        time_embedding = time_embedding.to(device=z_t.device, dtype=h.dtype)
        h = h + self.time_mlp(time_embedding).to(dtype=h.dtype)[:, None, :]

        h = self.transformer(h, src_key_padding_mask=~mask)
        h = self.output_norm(h)
        return self.output_proj(h)


class DiTBlock(nn.Module):
    """A single DiT block: adaLN-Zero modulated self-attention and MLP."""

    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = AdaLayerNormZero(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, emb=emb)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out
        normed = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(normed)
        return x


class DiTFlowTransformer(nn.Module):
    """Minimal DiT backbone for latent flow matching with adaLN-Zero time conditioning."""

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        if config.model_dim % config.num_heads != 0:
            raise ValueError("diffusion.model_dim must be divisible by diffusion.num_heads.")
        if config.objective != "flow_matching":
            raise ValueError(f"Unsupported diffusion objective: {config.objective}")
        if config.path not in ("linear", "cosine"):
            raise ValueError(f"Unsupported diffusion path: {config.path}")
        if config.t_sampling != "uniform":
            raise ValueError(f"Unsupported t_sampling: {config.t_sampling}")

        self.config = config
        # No input LayerNorm: it erases the magnitude of z_t, which the velocity target
        # depends on. Latents are expected to be standardized (see latent_stats_path).
        self.input_proj = nn.Linear(config.latent_dim, config.model_dim)
        if config.self_conditioning:
            self.self_cond_proj = nn.Linear(config.latent_dim, config.model_dim)
        # Prefix-conditioning "flag" mode: a learned marker added to positions holding
        # known-clean latents, so the model is told (not asked to detect) which inputs
        # are ground truth. Zero-init keeps the flag-off forward identical at init.
        if getattr(config, "prefix_cond_enabled", False) and getattr(config, "prefix_cond_mode", "data") == "flag":
            self.clean_flag = nn.Parameter(torch.zeros(config.model_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_words, config.model_dim))

        # Continuous t in [0, 1] scaled by 1000 to match the legacy frequency range.
        self.time_proj = Timesteps(config.model_dim, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.time_embed = TimestepEmbedding(config.model_dim, config.model_dim)

        self.blocks = nn.ModuleList(
            DiTBlock(config.model_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        )

        self.output_norm = nn.LayerNorm(config.model_dim, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.model_dim, 2 * config.model_dim))
        self.output_proj = nn.Linear(config.model_dim, config.latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        # adaLN-Zero: start every modulation at zero so blocks and the final layer are identity.
        for block in self.blocks:
            nn.init.zeros_(block.norm1.linear.weight)
            nn.init.zeros_(block.norm1.linear.bias)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # Zero-init self-conditioning so enabling it starts as a no-op.
        if self.config.self_conditioning:
            nn.init.zeros_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        z_self: torch.Tensor | None = None,
        clean_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_t.ndim != 3:
            raise ValueError(f"Expected z_t with shape [batch, words, latent_dim], got {tuple(z_t.shape)}.")
        batch_size, word_count, latent_dim = z_t.shape
        if latent_dim != self.config.latent_dim:
            raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {latent_dim}.")
        if word_count > self.config.max_words:
            raise ValueError(f"word_count={word_count} exceeds max_words={self.config.max_words}.")

        if mask is None:
            mask = torch.ones((batch_size, word_count), device=z_t.device, dtype=torch.bool)
        else:
            mask = mask.to(device=z_t.device, dtype=torch.bool)

        h = self.input_proj(z_t)
        if self.config.self_conditioning:
            if z_self is None:
                z_self = torch.zeros_like(z_t)
            h = h + self.self_cond_proj(z_self)
        if clean_mask is not None:
            # [B, W] bool marking positions whose z_t holds the known-clean latent.
            clean = clean_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
            h = h + clean * self.clean_flag
        h = h + self.position_embedding[:, :word_count, :].to(dtype=h.dtype)

        emb = self.time_proj(t.to(z_t.device).flatten()).to(dtype=h.dtype)
        emb = self.time_embed(emb)

        key_padding_mask = ~mask
        for block in self.blocks:
            h = block(h, emb, key_padding_mask)

        shift, scale = self.output_modulation(emb).chunk(2, dim=-1)
        h = self.output_norm(h) * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.output_proj(h)


class PlainDiTLatentPrior(nn.Module):
    """Stock-DiT shaped backbone over word latents arranged as a square image.

    The model keeps DiT's patch embed, 2D sin-cos positional embedding, dummy class
    conditioning, explicit adaLN-Zero blocks, final layer, and unpatchify path. The
    adapter only reshapes [B, words, latent_dim] to [B, latent_dim, H, W].
    """

    patch_size = 1

    def __init__(self, config: DiffusionModelConfig):
        super().__init__()
        if config.model_dim % config.num_heads != 0:
            raise ValueError("diffusion.model_dim must be divisible by diffusion.num_heads.")
        if config.objective != "flow_matching":
            raise ValueError(f"Unsupported diffusion objective: {config.objective}")
        if config.path not in ("linear", "cosine"):
            raise ValueError(f"Unsupported diffusion path: {config.path}")
        if config.t_sampling != "uniform":
            raise ValueError(f"Unsupported t_sampling: {config.t_sampling}")

        grid_size = math.isqrt(config.max_words)
        if grid_size * grid_size != config.max_words:
            raise ValueError(
                f"plain_dit requires diffusion.max_words to be a perfect square, got {config.max_words}."
            )

        self.config = config
        self.grid_size = grid_size
        self.in_channels = config.latent_dim
        self.out_channels = config.latent_dim
        self.x_embedder = PlainPatchEmbed(grid_size, self.patch_size, config.latent_dim, config.model_dim)
        # Optional self-conditioning: a second patch-embed for the fed-back x̂₀, zero-initialized
        # (see initialize_weights) so enabling it starts as a no-op -- mirrors the dit backbone's
        # self_cond_proj. The fed-back estimate enters the stock DiT image the same way the input does.
        if config.self_conditioning:
            self.x_embedder_self = PlainPatchEmbed(grid_size, self.patch_size, config.latent_dim, config.model_dim)
        # Optional context channel (prefix_cond_mode == "channel"): a separate zero-init patch-embed
        # carrying clean context latents plus a zero-init learned marker (context_flag) announcing
        # which positions are context. Both start as a no-op so a checkpoint trained without them
        # loads bit-for-bit (see initialize_weights).
        if uses_context_channel(config):
            self.x_embedder_ctx = PlainPatchEmbed(grid_size, self.patch_size, config.latent_dim, config.model_dim)
            self.context_flag = nn.Parameter(torch.zeros(config.model_dim))
        self.t_embedder = PlainTimestepEmbedder(config.model_dim)
        self.y_embedder = PlainLabelEmbedder(config.model_dim)
        # Frozen 2D sin-cos by default (stock DiT); learn_pos_embed unfreezes it (values
        # still initialized from sin-cos in initialize_weights, so warm starts are exact).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, config.model_dim),
            requires_grad=getattr(config, "learn_pos_embed", False),
        )
        self.blocks = nn.ModuleList(
            PlainDiTBlock(config.model_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        )
        self.final_layer = PlainFinalLayer(config.model_dim, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        if self.config.self_conditioning:  # zero-init so enabling self-cond starts as a no-op
            nn.init.zeros_(self.x_embedder_self.proj.weight)
            nn.init.zeros_(self.x_embedder_self.proj.bias)
        if uses_context_channel(self.config):  # zero-init so enabling the context channel starts as a no-op
            nn.init.zeros_(self.x_embedder_ctx.proj.weight)
            nn.init.zeros_(self.x_embedder_ctx.proj.bias)
            # context_flag is already zeros from its constructor.
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Expected a square token grid, got {x.shape[1]} tokens.")
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def _pad_to_canvas(self, z_t: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, word_count, latent_dim = z_t.shape
        if word_count > self.config.max_words:
            raise ValueError(f"word_count={word_count} exceeds max_words={self.config.max_words}.")
        if latent_dim != self.config.latent_dim:
            raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {latent_dim}.")
        if word_count == self.config.max_words:
            return z_t, mask

        padded = z_t.new_zeros((batch_size, self.config.max_words, latent_dim))
        padded_mask = torch.zeros((batch_size, self.config.max_words), device=z_t.device, dtype=torch.bool)
        padded[:, :word_count] = z_t
        padded_mask[:, :word_count] = mask
        return padded, padded_mask

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        z_self: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_t.ndim != 3:
            raise ValueError(f"Expected z_t with shape [batch, words, latent_dim], got {tuple(z_t.shape)}.")

        batch_size, word_count, _ = z_t.shape
        if mask is None:
            mask = torch.ones((batch_size, word_count), device=z_t.device, dtype=torch.bool)
        else:
            mask = mask.to(device=z_t.device, dtype=torch.bool)
        z_canvas, canvas_mask = self._pad_to_canvas(z_t, mask)

        # Original DiT has no attention mask; padded word slots are made blank and
        # are excluded by the caller's masked loss after reshaping back.
        z_canvas = z_canvas * canvas_mask.unsqueeze(-1).to(dtype=z_canvas.dtype)
        x = z_canvas.transpose(1, 2).reshape(batch_size, self.config.latent_dim, self.grid_size, self.grid_size)
        x = self.x_embedder(x) + self.pos_embed.to(dtype=x.dtype)
        if self.config.self_conditioning:
            if z_self is None:
                z_self = torch.zeros_like(z_t)
            z_self_canvas, _ = self._pad_to_canvas(z_self, mask)
            z_self_canvas = z_self_canvas * canvas_mask.unsqueeze(-1).to(dtype=z_self_canvas.dtype)
            xs = z_self_canvas.transpose(1, 2).reshape(batch_size, self.config.latent_dim, self.grid_size, self.grid_size)
            x = x + self.x_embedder_self(xs)
        # Context channel: add the context embedding and the learned marker ONLY at context
        # positions. The 1x1 patch-embed is per-position, so masking its output is exact -- a
        # non-context token (and an all-null / dropped-out row) receives exactly zero here, so
        # the null branch is bit-identical to the unconditional model. context=None skips it,
        # which is equivalent since the masked contribution would be zero anyway.
        if uses_context_channel(self.config) and context is not None:
            if context_mask is None:
                raise ValueError("context requires context_mask in channel mode.")
            ctx_canvas, cm_canvas = self._pad_to_canvas(context, context_mask.to(dtype=torch.bool))
            xc = ctx_canvas.transpose(1, 2).reshape(batch_size, self.config.latent_dim, self.grid_size, self.grid_size)
            cm = cm_canvas.unsqueeze(-1).to(dtype=x.dtype)
            x = x + self.x_embedder_ctx(xc) * cm
            x = x + cm * self.context_flag
        t_emb = self.t_embedder(t.to(z_t.device).flatten() * 1000.0).to(dtype=x.dtype)
        labels = torch.zeros(batch_size, device=z_t.device, dtype=torch.long)
        y_emb = self.y_embedder(labels).to(dtype=x.dtype)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        x = x.reshape(batch_size, self.config.latent_dim, self.config.max_words).transpose(1, 2)
        return x[:, :word_count]

    def warmstart_allowed_missing(self) -> set[str]:
        """State-dict keys permitted to be absent when warm-starting (init_from_checkpoint) from a
        checkpoint that predates the zero-initialized context channel. Empty unless channel mode is
        on, so the load stays a strict-except-these-exact-keys check (see utils.load_checkpoint)."""
        if uses_context_channel(self.config):
            return {"x_embedder_ctx.proj.weight", "x_embedder_ctx.proj.bias", "context_flag"}
        return set()


def build_diffusion_backbone(config: DiffusionModelConfig) -> nn.Module:
    """Construct the diffusion backbone selected by ``config.backbone``."""

    if (
        getattr(config, "prefix_cond_enabled", False)
        and getattr(config, "prefix_cond_mode", "data") == "flag"
        and config.backbone != "dit"
    ):
        raise ValueError("prefix_cond_mode='flag' requires backbone='dit' (clean_mask marker).")
    if uses_context_channel(config) and config.backbone != "plain_dit":
        raise ValueError("prefix_cond_mode='channel' requires backbone='plain_dit' (context patch-embed).")
    if config.backbone == "dit":
        return DiTFlowTransformer(config)
    if config.backbone == "plain_dit":
        return PlainDiTLatentPrior(config)
    if config.backbone == "transformer":
        return LatentFlowTransformer(config)
    raise ValueError(
        f"Unknown diffusion.backbone: {config.backbone!r} "
        "(expected 'transformer', 'dit', or 'plain_dit')."
    )
