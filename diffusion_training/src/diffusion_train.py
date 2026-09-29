import dataclasses
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from src.diffusion_model import LatentFlowTransformer, build_diffusion_backbone
from src.model import DiffusionAutoencoder
from src.train import (
    MasterWeightAdamW,
    _dtype_from_name,
    apply_tf32,
    build_training_loader,
    build_validation_loaders,
    cycle_batches,
    diffusion_dtype_from_config,
    latent_isotropy_loss,
    learning_rate_for_step,
    move_batch_to_device,
    parameter_groups,
)
from src.utils import (
    DiffusionExperimentConfig,
    create_logger,
    load_checkpoint,
    module_dtype,
    save_checkpoint,
    seed_everything,
)


@dataclass
class DiffusionTrainResult:
    final_step: int
    trainable_parameters: int
    checkpoint_paths: list[str] = field(default_factory=list)
    validation_losses: dict[str, float] = field(default_factory=dict)
    logger_url: str | None = None


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_optimizer(model: torch.nn.Module, config: DiffusionExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameter_groups(model, config.training.weight_decay),
        lr=config.training.lr,
        betas=config.training.betas,
    )


def load_frozen_autoencoder(config: DiffusionExperimentConfig, device: torch.device) -> DiffusionAutoencoder:
    compression_dim = config.model.latent_compression_dim
    if compression_dim is not None and compression_dim != config.diffusion.latent_dim:
        raise ValueError(
            "model.latent_compression_dim "
            f"({compression_dim}) must match diffusion.latent_dim ({config.diffusion.latent_dim})."
        )
    autoencoder = DiffusionAutoencoder.from_pretrained(
        config.model,
        torch_dtype=_dtype_from_name(config.training.dtype),
        device=device,
        freeze_encoder=True,
        local_files_only=config.autoencoder.local_files_only,
    )
    if config.autoencoder.checkpoint:
        load_checkpoint(config.autoencoder.checkpoint, model=autoencoder, map_location=device)
    if config.autoencoder.freeze:
        autoencoder.requires_grad_(False)
    autoencoder.eval()
    return autoencoder


def load_latent_stats(
    path: str,
    device: torch.device,
    latent_dim: int,
) -> dict[str, torch.Tensor]:
    """Load the per-dim latent mean/std artifact written by scripts/compute_latent_stats.py."""

    stats = torch.load(path, map_location=device)
    mean = stats["mean"].to(device=device, dtype=torch.float32).reshape(-1)
    std = stats["std"].to(device=device, dtype=torch.float32).reshape(-1).clamp_min(1e-3)
    if mean.numel() != latent_dim or std.numel() != latent_dim:
        raise ValueError(
            f"latent stats dim {mean.numel()} does not match diffusion.latent_dim {latent_dim} ({path})"
        )
    return {"mean": mean, "std": std}


def _encode_latent_batch_impl(
    autoencoder: torch.nn.Module,
    batch: dict[str, Any],
    max_words: int,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    latent_stats: dict[str, torch.Tensor] | None = None,
    return_raw: bool = False,
) -> dict[str, torch.Tensor]:
    """Shared encode body. Grad/eval policy belongs to the wrappers below, not here."""
    z_words = autoencoder.encode(batch["byte_ids"], batch["word_boundaries"])
    batch_size = len(z_words)
    z_data = torch.zeros((batch_size, max_words, latent_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, max_words), device=device, dtype=torch.bool)
    lengths = torch.zeros(batch_size, device=device, dtype=torch.long)

    for index, z_item in enumerate(z_words):
        if z_item.ndim == 3:
            z_item = z_item.squeeze(0)
        if z_item.ndim != 2:
            raise ValueError(f"Expected encoded latents with shape [words, dim], got {tuple(z_item.shape)}.")
        if z_item.size(-1) != latent_dim:
            raise ValueError(f"Expected latent_dim={latent_dim}, got {z_item.size(-1)}.")

        length = min(int(z_item.size(0)), max_words)
        if length <= 0:
            continue
        z_data[index, :length] = z_item[:length].to(device=device, dtype=dtype)
        mask[index, :length] = True
        lengths[index] = length

    if latent_stats is not None:
        mean = latent_stats["mean"].to(dtype=z_data.dtype)
        std = latent_stats["std"].to(dtype=z_data.dtype)
        z_data = (z_data - mean) / std
        z_data = z_data * mask.unsqueeze(-1).to(dtype=z_data.dtype)

    out = {"z": z_data, "mask": mask, "lengths": lengths}
    if return_raw:
        # Native-space latents as the encoder emitted them (pre-standardization,
        # pre-truncation): the isotropy penalty's contract is the RAW space.
        out["z_raw"] = z_words
    return out


@torch.no_grad()
def encode_latent_batch(
    autoencoder: torch.nn.Module,
    batch: dict[str, Any],
    max_words: int,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    latent_stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    autoencoder.eval()
    return _encode_latent_batch_impl(autoencoder, batch, max_words, latent_dim, dtype, device, latent_stats)


def encode_latent_batch_trainable(
    autoencoder: torch.nn.Module,
    batch: dict[str, Any],
    max_words: int,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    latent_stats: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Joint-encoder encode: gradients flow into the encoder; the caller owns train/eval
    modes (encoder side in train mode, decoder side eval -- see the training loop)."""
    return _encode_latent_batch_impl(
        autoencoder, batch, max_words, latent_dim, dtype, device, latent_stats, return_raw=True
    )


def sample_uniform_t(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)


# Noise bands for train/mse_by_logsnr/*. On the linear path lambda = 2*log(t/(1-t)), so the
# edges -4/-1/1/4 fall at t ~ 0.12 / 0.38 / 0.62 / 0.88: roughly "near-noise", "noisy",
# "mid", "clean-ish", "near-clean".
LOGSNR_BINS: tuple[tuple[float, float, str], ...] = (
    (float("-inf"), -4.0, "lt-4"),
    (-4.0, -1.0, "-4to-1"),
    (-1.0, 1.0, "-1to1"),
    (1.0, 4.0, "1to4"),
    (4.0, float("inf"), "gt4"),
)


def path_coefficients(t, path: str = "linear", timeshift: float | None = 1.0):
    if timeshift is not None and timeshift != 1.0:
        if timeshift <= 0.0:
            raise ValueError(f"path_timeshift must be > 0, got {timeshift}")
        denom = 1.0 + (timeshift - 1.0) * t
        t = timeshift * t / denom
        dtau = timeshift / (denom * denom)
    else:
        dtau = 1.0
    if path == "linear":
        if torch.is_tensor(t):
            one = torch.ones_like(t)
            return t, 1.0 - t, one * dtau, -one * dtau
        return t, 1.0 - t, dtau, -dtau
    if path == "cosine":
        half_pi = math.pi / 2.0
        if torch.is_tensor(t):
            alpha = torch.sin(half_pi * t)
            sigma = torch.cos(half_pi * t)
            return alpha, sigma, half_pi * sigma * dtau, -half_pi * alpha * dtau
        alpha = math.sin(half_pi * t)
        sigma = math.cos(half_pi * t)
        return alpha, sigma, half_pi * sigma * dtau, -half_pi * alpha * dtau
    raise ValueError(f"Unsupported diffusion path: {path}")


def velocity_to_x0(z_t, velocity, t, path: str = "linear", timeshift: float | None = 1.0):
    """Clean-latent estimate x̂₀ implied by a predicted velocity on the given path.

    Inverts ``z_t = alpha*x0 + sigma*noise`` and ``velocity = alpha_dot*x0 + sigma_dot*noise``.
    For the linear path this reduces to ``x0 = z_t + (1 - t) * velocity``.
    """
    alpha, sigma, alpha_dot, sigma_dot = path_coefficients(t, path, timeshift)
    alpha, sigma, alpha_dot, sigma_dot = (c[:, None, None] for c in (alpha, sigma, alpha_dot, sigma_dot))
    return (sigma * velocity - sigma_dot * z_t) / (sigma * alpha_dot - sigma_dot * alpha)


def x0_to_velocity(z_t, x0, t, path: str = "linear", timeshift: float | None = 1.0):
    alpha, sigma, alpha_dot, sigma_dot = path_coefficients(t, path, timeshift)
    alpha, sigma, alpha_dot, sigma_dot = (c[:, None, None] for c in (alpha, sigma, alpha_dot, sigma_dot))
    eps = (z_t - alpha * x0) / sigma.clamp_min(1e-4)  # guard sigma -> 0 as t -> 1
    # path_coefficients are float32; cast back so the Euler update stays in the model dtype.
    return (alpha_dot * x0 + sigma_dot * eps).to(x0.dtype)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_position = (prediction.float() - target.float()).pow(2).mean(dim=-1)
    mask_f = mask.to(dtype=per_position.dtype)
    return (per_position * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def sample_prefix_mask(mask: torch.Tensor, prob: float) -> torch.Tensor:
    """[B, W] bool; True = position held clean (prefix-conditioning).

    Per row: with probability ``prob`` and valid length len_i >= 2, prefix length
    P_i ~ UniformInt[1, len_i - 1], so at least one position always stays in the loss.
    Relies on ``mask`` being contiguous-prefix True (encode_latent_batch guarantees it);
    padding positions therefore always come out False.
    """
    lengths = mask.sum(dim=1)
    eligible = (lengths >= 2) & (torch.rand(lengths.shape, device=mask.device) < prob)
    # UniformInt[1, len-1]; clamp keeps ineligible rows in-range (they are zeroed below).
    prefix_lengths = (torch.rand(lengths.shape, device=mask.device) * (lengths - 1).clamp_min(1)).long() + 1
    prefix_lengths = torch.where(eligible, prefix_lengths, torch.zeros_like(prefix_lengths))
    positions = torch.arange(mask.size(1), device=mask.device)
    return positions[None, :] < prefix_lengths[:, None]


def decoder_ce_anchor(
    autoencoder: torch.nn.Module,
    x0_hat_std: torch.Tensor,
    ce_context: dict[str, Any],
    latent_stats: dict[str, torch.Tensor] | None,
    x0_clamp: float,
    dtype: torch.dtype,
    prefix_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    byte_ids = ce_context["byte_ids"]
    labels = ce_context["labels"].clone()
    boundaries = ce_context["word_boundaries"]
    lengths = ce_context["lengths"]
    device = x0_hat_std.device

    if x0_clamp and x0_clamp > 0:
        x0_hat_std = x0_hat_std.clamp(-x0_clamp, x0_clamp)
    mean = latent_stats["mean"] if latent_stats is not None else None
    std = latent_stats["std"] if latent_stats is not None else None

    rows: list[int] = []
    z_list: list[torch.Tensor] = []
    boundary_list: list[torch.Tensor] = []
    for index in range(x0_hat_std.size(0)):
        length = int(lengths[index].item())
        if length <= 0:
            continue
        boundary = boundaries[index].to(device=device, dtype=torch.long)
        byte_cut = int(boundary[length].item())
        z = x0_hat_std[index, :length].float()
        if std is not None:
            z = z * std + mean
        z_list.append(z.to(dtype=dtype).unsqueeze(0))
        boundary_list.append(boundary[: length + 1])
        # Drop the transition byte (and any bytes of truncated-off words): its next-byte target
        # belongs to a word whose latent was not fed to the decoder, so it is not predictable here.
        labels[index, max(0, byte_cut - 1) :] = -100
        # Prefix-conditioning: clean-context words are excluded from supervision; the label at
        # byte_prefix - 1 targets the first byte of the first continuation word, so it is kept.
        prefix_words = int(prefix_lengths[index].item()) if prefix_lengths is not None else 0
        if prefix_words > 0:
            byte_prefix = int(boundary[prefix_words].item())
            labels[index, : max(0, byte_prefix - 1)] = -100
        rows.append(index)

    if not rows:
        return x0_hat_std.new_zeros(())

    row_index = torch.tensor(rows, device=device)
    output = autoencoder.forward_from_latents(
        byte_ids=byte_ids[row_index],
        z_words=z_list,
        word_boundaries=boundary_list,
        labels=labels[row_index],
    )
    return output["loss"]


def flow_matching_loss(
    model: LatentFlowTransformer,
    z_data: torch.Tensor,
    mask: torch.Tensor,
    autoencoder: torch.nn.Module | None = None,
    ce_context: dict[str, Any] | None = None,
    latent_stats: dict[str, torch.Tensor] | None = None,
    t_power: float | None = None,
    prefix_cond: bool | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    noise = torch.randn_like(z_data)
    t = sample_uniform_t(z_data.size(0), z_data.device)
    if t_power is None:
        t_power = getattr(model.config, "t_power", 1.0)
    if t_power != 1.0:
        t = t**t_power
    path = getattr(model.config, "path", "linear")
    timeshift = getattr(model.config, "path_timeshift", 1.0)
    alpha, sigma, alpha_dot, sigma_dot = path_coefficients(t, path, timeshift)
    # Per-sample log-SNR of the interpolant, lambda = log(alpha^2/sigma^2). Latents are
    # standardized (unit variance), so this nominal schedule SNR is the actual one. Unlike t,
    # lambda is comparable across (path, path_timeshift, t_power) recipes. Clamp bounds it to
    # ~[-27.6, 27.6] at the endpoints, where the true value is +-inf.
    logsnr = 2.0 * (alpha.float().clamp_min(1e-6).log() - sigma.float().clamp_min(1e-6).log())
    alpha = alpha[:, None, None].to(dtype=z_data.dtype)
    sigma = sigma[:, None, None].to(dtype=z_data.dtype)
    alpha_dot = alpha_dot[:, None, None].to(dtype=z_data.dtype)
    sigma_dot = sigma_dot[:, None, None].to(dtype=z_data.dtype)
    z_t = alpha * z_data + sigma * noise
    prediction_mode = getattr(model.config, "prediction", "velocity")
    target = z_data if prediction_mode == "x0" else alpha_dot * z_data + sigma_dot * noise

    # Prefix-conditioning: substitute clean latents on a sampled prefix and drop those
    # positions from the loss. Everything (incl. RNG draws) stays inside this guard so
    # disabled runs consume an identical RNG stream.
    if prefix_cond is None:
        prefix_cond = getattr(model.config, "prefix_cond_enabled", False)
    prefix_mask = None
    cond_kwargs: dict[str, torch.Tensor] = {}
    if prefix_cond:
        mode = getattr(model.config, "prefix_cond_mode", "data")
        if mode not in ("data", "flag", "channel"):
            raise NotImplementedError(
                f"prefix_cond_mode={mode!r} is reserved; 'data', 'flag', and 'channel' are implemented."
            )
        prefix_mask = sample_prefix_mask(mask, getattr(model.config, "prefix_cond_prob", 0.5))
        if mode == "channel":
            # Separate channel: z_t stays PURE NOISE at context positions (no substitution). The
            # true (optionally corruption-augmented) context latents are passed on their own channel.
            context = z_data
            context_noise = getattr(model.config, "prefix_cond_context_noise", 0.0)
            if context_noise > 0:
                # t_ctx ~ U(1 - delta, 1); t=1 is clean, t=0 is pure noise (path_coefficients).
                t_ctx = 1.0 - context_noise * torch.rand(z_data.size(0), device=z_data.device)
                a_ctx, s_ctx, _, _ = path_coefficients(t_ctx, path, timeshift)
                a_ctx = a_ctx[:, None, None].to(dtype=z_data.dtype)
                s_ctx = s_ctx[:, None, None].to(dtype=z_data.dtype)
                context = a_ctx * z_data + s_ctx * torch.randn_like(z_data)
            dropout = getattr(model.config, "prefix_cond_dropout", 0.0)
            if dropout > 0:
                # Blank the channel on whole rows -> the unconditional (CFG null) branch. keep is
                # per-row, so surviving rows keep their contiguous prefix mask.
                keep = torch.rand(z_data.size(0), device=z_data.device) >= dropout
                prefix_mask = prefix_mask & keep[:, None]
            cond_kwargs = {"context": context, "context_mask": prefix_mask}
        else:
            z_t = torch.where(prefix_mask.unsqueeze(-1), z_data, z_t)
            if mode == "flag":
                # Mark the clean positions in the model input (learned marker); kwargs so the
                # data-mode call signature stays untouched for all backbones.
                cond_kwargs = {"clean_mask": prefix_mask}

    z_self = None
    if getattr(model.config, "self_conditioning", False) and torch.rand(()) < 0.5:
        with torch.no_grad():
            out0 = model(z_t, t, mask, **cond_kwargs)
            # x0 mode emits x̂₀ directly; velocity mode derives it from the predicted velocity.
            x0_self = out0 if prediction_mode == "x0" else velocity_to_x0(z_t, out0, t, path, timeshift)
            if prefix_mask is not None:
                # The prefix is known-clean; feed the truth back, matching sampling time.
                x0_self = torch.where(prefix_mask.unsqueeze(-1), z_data, x0_self)
            z_self = x0_self.to(z_data.dtype)
    prediction = model(z_t, t, mask, z_self, **cond_kwargs)
    loss_mask = mask if prefix_mask is None else mask & ~prefix_mask
    mse = masked_mse(prediction, target, loss_mask)
    loss = mse

    # Per-row MSE paired with each row's log-SNR, so the trainer can log the training loss
    # binned by noise level (an aggregate loss can hide regressions in one noise band).
    with torch.no_grad():
        per_position = (prediction.float() - target.float()).pow(2).mean(dim=-1)
        row_weight = loss_mask.to(per_position.dtype)
        row_counts = row_weight.sum(dim=1)
        row_mse = (per_position * row_weight).sum(dim=1) / row_counts.clamp_min(1.0)
        supervised = row_counts > 0

    ce_value = 0.0
    ce_clip_frac = 0.0
    if (
        getattr(model.config, "ce_anchor_enabled", False)
        and autoencoder is not None
        and ce_context is not None
    ):
        # The decoder consumes a clean-latent estimate. x0 mode predicts x̂₀ directly; velocity
        # mode converts. The conversion is exact and differentiable, and needs no numerical guard:
        # its denominator (sigma*alpha_dot - sigma_dot*alpha) is constant on both paths -- dtau for
        # linear, (pi/2)*dtau for cosine -- so it never approaches zero. It scales the CE gradient
        # reaching the prediction by sigma(t)/denom: full strength at t->0, tapering to 0 at t->1,
        # where x̂₀ ≈ z_t no matter what the model predicts and CE carries no signal about it.
        x0_for_ce = (
            prediction if prediction_mode == "x0" else velocity_to_x0(z_t, prediction, t, path, timeshift)
        )
        if prefix_mask is not None:
            # Decoder must cross-attend to TRUE context latents, and CE must not backprop
            # into loss-excluded prefix positions. MUST come after the conversion above: in
            # data/flag mode z_t already holds clean latents on the prefix, so velocity_to_x0
            # produces nonsense there and this overwrite is what removes it.
            x0_for_ce = torch.where(prefix_mask.unsqueeze(-1), z_data, x0_for_ce)
        x0_clamp = getattr(model.config, "ce_anchor_x0_clamp", 0.0)
        if x0_clamp and x0_clamp > 0 and bool(loss_mask.any()):
            # Fraction of supervised positions where the clamp bites (any latent dim over it).
            # ce_anchor_x0_clamp was calibrated against x0-mode predictions; velocity-mode x̂₀
            # carries an extra sigma(t)/denom factor, so this says whether the clamp is inert
            # or dominating before it silently reshapes the CE gradient.
            with torch.no_grad():
                over = (x0_for_ce.detach().abs() > x0_clamp).any(dim=-1)
                ce_clip_frac = float(over[loss_mask].float().mean().item())
        ce = decoder_ce_anchor(
            autoencoder,
            x0_for_ce,
            ce_context,
            latent_stats,
            x0_clamp=x0_clamp,
            # The decoder's dtype, not the latent's: under training.diffusion_dtype the DiT
            # (and so x0_for_ce) can be float32 while the frozen autoencoder stays bfloat16.
            dtype=module_dtype(autoencoder, z_data.dtype),
            prefix_lengths=prefix_mask.sum(dim=1) if prefix_mask is not None else None,
        )
        loss = mse + getattr(model.config, "ce_anchor_weight", 0.0) * ce
        ce_value = float(ce.item())

    valid_z = z_data[mask]
    valid_noise = noise[mask]
    metrics = {
        "t_mean": float(t.mean().item()),
        "z_norm": float(valid_z.float().norm(dim=-1).mean().item()) if valid_z.numel() else 0.0,
        "noise_norm": float(valid_noise.float().norm(dim=-1).mean().item()) if valid_noise.numel() else 0.0,
        "mse": float(mse.item()),
        "ce": ce_value,
        "ce_clip_frac": ce_clip_frac,
        "prefix_frac": float(prefix_mask[mask].float().mean().item()) if prefix_mask is not None else 0.0,
        # Length/cutoff telemetry (logged as distributions by the trainer): valid words
        # per sequence, and -- when conditioning is on -- the sampled context cutoff per
        # CONDITIONED row plus the conditioned-row count for cond/conditioned_frac.
        "seq_lens": mask.sum(dim=1).detach().cpu().tolist(),
        "num_rows": int(mask.size(0)),
        # Per-row (log-SNR, MSE) pairs for rows with at least one supervised position; the
        # trainer logs the lambda distribution and the loss binned by noise band.
        "row_logsnr": logsnr[supervised].cpu().tolist(),
        "row_mse": row_mse[supervised].cpu().tolist(),
    }
    if prefix_mask is not None:
        prefix_lens = prefix_mask.sum(dim=1)
        metrics["prefix_lens"] = prefix_lens[prefix_lens > 0].detach().cpu().tolist()
        metrics["num_conditioned"] = int((prefix_lens > 0).sum().item())
    return loss, metrics


@torch.no_grad()
def evaluate_per_slice(
    model: LatentFlowTransformer,
    autoencoder: torch.nn.Module,
    validation_loaders: dict[str, Iterable[dict[str, Any]]],
    config: DiffusionExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    latent_stats: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-slice flow MSE plus training-health metrics (per-t loss vs the linear
    baseline, and statistics of Euler-sampled latents vs real latents)."""

    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}
    health_batches: list[dict[str, torch.Tensor]] = []
    for name, loader in validation_loaders.items():
        total_loss = 0.0
        total_batches = 0
        for batch_index, batch in enumerate(loader):
            if batch_index >= config.training.validation_batches_per_slice:
                break
            batch = move_batch_to_device(batch, device)
            latent_batch = encode_latent_batch(
                autoencoder,
                batch,
                max_words=config.diffusion.max_words,
                latent_dim=config.diffusion.latent_dim,
                dtype=dtype,
                device=device,
                latent_stats=latent_stats,
            )
            # Keep validation on uniform t and unconditional so the flow-MSE stays a stable,
            # recipe-independent yardstick (comparable across prefix-conditioning arms).
            loss, _ = flow_matching_loss(model, latent_batch["z"], latent_batch["mask"], t_power=1.0, prefix_cond=False)
            total_loss += float(loss.item())
            total_batches += 1
            if len(health_batches) < 2:
                health_batches.append(latent_batch)
        if total_batches > 0:
            losses[name] = total_loss / total_batches
    health = health_metrics(model, health_batches, config, device, dtype)
    if was_training:
        model.train()
    return losses, health


@torch.no_grad()
def health_metrics(
    model: torch.nn.Module,
    latent_batches: list[dict[str, torch.Tensor]],
    config: DiffusionExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    per_t_values: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    generator = torch.Generator(device=device).manual_seed(1234)
    path = config.diffusion.path
    timeshift = getattr(config.diffusion, "path_timeshift", 1.0)
    prediction_mode = getattr(config.diffusion, "prediction", "velocity")

    # Per-t loss vs the path-optimal scalar-gain baseline. NOTE: with a path timeshift,
    # a given t maps to a different noise level, so per-t values are only comparable
    # across runs sharing the same (path, path_timeshift).
    for t_value in per_t_values:
        alpha, sigma, alpha_dot, sigma_dot = path_coefficients(t_value, path, timeshift)
        # Best scalar g minimizing E||target - g*z_t||^2 for unit-variance data:
        # g = (alpha_dot*alpha + sigma_dot*sigma) / (alpha^2 + sigma^2). For the
        # linear path this reduces to (2t-1)/((1-t)^2 + t^2).
        gain = (alpha_dot * alpha + sigma_dot * sigma) / (alpha**2 + sigma**2)
        model_total, baseline_total, batches = 0.0, 0.0, 0
        for latent_batch in latent_batches:
            z, mask = latent_batch["z"], latent_batch["mask"]
            noise = torch.randn(z.shape, generator=generator, device=device, dtype=z.dtype)
            z_t = alpha * z + sigma * noise
            target = alpha_dot * z + sigma_dot * noise
            t = torch.full((z.size(0),), t_value, device=device)
            raw = model(z_t, t, mask)
            # Score in velocity space so the linear-gain baseline stays comparable across modes.
            prediction = x0_to_velocity(z_t, raw, t, path, timeshift) if prediction_mode == "x0" else raw
            model_total += float(masked_mse(prediction, target, mask).item())
            baseline_total += float(masked_mse(gain * z_t, target, mask).item())
            batches += 1
        if batches:
            metrics[f"loss_t{t_value:.2f}"] = model_total / batches
            metrics[f"linear_baseline_t{t_value:.2f}"] = baseline_total / batches
            # The noise level this probe actually sits at: with a path timeshift the same t
            # maps to a different lambda, so this is the axis to line runs up on.
            metrics[f"logsnr_t{t_value:.2f}"] = 2.0 * (
                math.log(max(alpha, 1e-6)) - math.log(max(sigma, 1e-6))
            )

    # Context-conditioning oracle probe: does clamping a clean first half reduce x0-space error on
    # the continuation? In-training version of the oracle-inpainting diagnostic
    # (notes/2026-07-04-continuation-quality-brainstorm.md). Run ALWAYS (even unconditional models)
    # so context_gain is a yardstick comparable across arms. The shuffled-context control feeds each
    # row ANOTHER document's context: if true and shuffled gains match, the model is exploiting the
    # clean half as a low-noise statistical cue, not as content, and the headline gain is worthless.
    mode = getattr(config.diffusion, "prefix_cond_mode", "data")
    prefix_on = getattr(config.diffusion, "prefix_cond_enabled", False)
    flag_mode = prefix_on and mode == "flag"
    channel_mode = prefix_on and mode == "channel"

    def _predict(z_t_uncond, z_ctx, half_mask, t):
        """Conditional (raw, z_t_used) for the active mode. Channel mode leaves z_t as pure noise
        and passes context on its own channel; data/flag substitute the context into z_t."""
        if channel_mode:
            return model(z_t_uncond, t, pool_mask, context=z_ctx, context_mask=half_mask), z_t_uncond
        z_t_cond = torch.where(half_mask.unsqueeze(-1), z_ctx, z_t_uncond)
        raw = model(z_t_cond, t, pool_mask, clean_mask=half_mask) if flag_mode else model(z_t_cond, t, pool_mask)
        return raw, z_t_cond

    def _to_x0(raw, z_t_used, t):
        return raw if prediction_mode == "x0" else velocity_to_x0(z_t_used, raw, t, path, timeshift)

    # Pool the collected health batches to a common word_count so the batch dim is >= 2 even at
    # validation_batch_size 1 -- required for the cross-document roll(1, dim=0) shuffle to shuffle.
    if latent_batches:
        w_max = max(int(b["z"].shape[1]) for b in latent_batches)
        d = config.diffusion.latent_dim
        z_parts, m_parts = [], []
        for b in latent_batches:
            zb, mb = b["z"], b["mask"]
            bsz, wb, _ = zb.shape
            if wb < w_max:
                zb = torch.cat([zb, zb.new_zeros(bsz, w_max - wb, d)], dim=1)
                mb = torch.cat([mb, mb.new_zeros((bsz, w_max - wb), dtype=torch.bool)], dim=1)
            z_parts.append(zb)
            m_parts.append(mb)
        pool_z = torch.cat(z_parts, dim=0)
        pool_mask = torch.cat(m_parts, dim=0)
        can_shuffle = pool_z.size(0) >= 2

        lengths = pool_mask.sum(dim=1)
        positions = torch.arange(pool_mask.size(1), device=pool_mask.device)
        half_mask = positions[None, :] < (lengths // 2)[:, None]
        cont_mask = pool_mask & ~half_mask
        z_shuf = pool_z.roll(1, dims=0)
        for t_value in (0.1, 0.3, 0.5, 0.7):
            alpha, sigma, _, _ = path_coefficients(t_value, path, timeshift)
            if not bool(cont_mask.any()):
                continue
            noise = torch.randn(pool_z.shape, generator=generator, device=device, dtype=pool_z.dtype)
            z_t = alpha * pool_z + sigma * noise
            t = torch.full((pool_z.size(0),), t_value, device=device)
            raw_uncond = model(z_t, t, pool_mask)
            x0_uncond = _to_x0(raw_uncond, z_t, t)
            raw_cond, zt_cond = _predict(z_t, pool_z, half_mask, t)
            x0_cond = _to_x0(raw_cond, zt_cond, t)
            uncond_mse = float(masked_mse(x0_uncond, pool_z, cont_mask).item())
            cond_mse = float(masked_mse(x0_cond, pool_z, cont_mask).item())
            metrics[f"prefix_uncond_mse_t{t_value:.2f}"] = uncond_mse
            metrics[f"prefix_cond_mse_t{t_value:.2f}"] = cond_mse
            denom = uncond_mse if uncond_mse > 0 else float("nan")
            metrics[f"context_gain_t{t_value:.2f}"] = 1.0 - cond_mse / denom
            if can_shuffle:
                raw_shuf, zt_shuf = _predict(z_t, z_shuf, half_mask, t)
                shuf_mse = float(masked_mse(_to_x0(raw_shuf, zt_shuf, t), pool_z, cont_mask).item())
                metrics[f"prefix_shuffled_mse_t{t_value:.2f}"] = shuf_mse
                metrics[f"context_gain_shuffled_t{t_value:.2f}"] = 1.0 - shuf_mse / denom

    # Euler-sample statistics vs real validation latents (standardized space).
    num_samples, num_words = 2, min(64, config.diffusion.max_words)
    z = torch.randn(
        (num_samples, num_words, config.diffusion.latent_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    sample_mask = torch.ones((num_samples, num_words), device=device, dtype=torch.bool)
    steps = config.diffusion.sampling_steps
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((num_samples,), (index + 0.5) / steps, device=device)
        raw = model(z, t, sample_mask)
        velocity = x0_to_velocity(z, raw, t, path, timeshift) if prediction_mode == "x0" else raw
        z = z + dt * velocity
    sampled = z.reshape(-1, config.diffusion.latent_dim).float()
    metrics["sampled_elem_std"] = float(sampled.std().item())
    metrics["sampled_norm_mean"] = float(sampled.norm(dim=-1).mean().item())
    metrics["sampled_norm_std"] = float(sampled.norm(dim=-1).std().item())

    real = [batch["z"][batch["mask"]].float() for batch in latent_batches]
    if real:
        real_cat = torch.cat(real, dim=0)
        if real_cat.size(0) > 1:
            metrics["real_elem_std"] = float(real_cat.std().item())
            metrics["real_norm_mean"] = float(real_cat.norm(dim=-1).mean().item())
            metrics["real_norm_std"] = float(real_cat.norm(dim=-1).std().item())
    return metrics


def train_diffusion(
    config: DiffusionExperimentConfig,
    autoencoder: torch.nn.Module | None = None,
    model: LatentFlowTransformer | None = None,
    train_loader: Iterable[dict[str, Any]] | None = None,
    validation_loaders: dict[str, Iterable[dict[str, Any]]] | None = None,
    device: str | torch.device | None = None,
) -> DiffusionTrainResult:
    seed_everything(config.training.seed)
    device = torch.device(device or config.training.device)
    apply_tf32(config.training)
    # The autoencoder keeps config.training.dtype (bfloat16; the vendored HAT stack requires
    # it). Everything the optimizer touches -- diffusion weights, latents, ODE state -- uses
    # diffusion_dtype, which defaults to the same value.
    dtype = diffusion_dtype_from_config(config.training)

    if autoencoder is None:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        autoencoder = load_frozen_autoencoder(config, device)
    else:
        autoencoder = autoencoder.to(device)
        autoencoder.requires_grad_(False)
        autoencoder.eval()

    latent_stats = None
    if config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(
            config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim
        )

    # Joint encoder training: the whole AE loads frozen exactly as before, then ONLY the
    # encoder side is re-enabled. The decoder stays a frozen teacher throughout -- that is
    # the leash keeping the drifting latents decodable (and the guard against the
    # decoder-as-LM degeneration measured in R2b-v1).
    joint = bool(config.training.joint_encoder)
    encoder_modules: list[torch.nn.Module] = []
    encoder_parameters: list[torch.nn.Parameter] = []
    if joint:
        if not config.autoencoder.freeze:
            raise ValueError(
                "training.joint_encoder expects autoencoder.freeze: true -- the AE loads fully "
                "frozen and only the encoder side is re-enabled here. freeze: false would leave "
                "decoder parameters accumulating gradients with no optimizer attached."
            )
        if config.training.encoder_lr is None or config.training.encoder_lr <= 0:
            raise ValueError("training.joint_encoder requires training.encoder_lr > 0 (no silent default).")
        if latent_stats is not None:
            stats_mean_max = float(latent_stats["mean"].abs().max().item())
            stats_std_dev = float((latent_stats["std"] - 1).abs().max().item())
            # The raw/standardized coherence constraint is load-bearing only when the
            # isotropy penalty (raw space) is active alongside the standardized flow;
            # iso-off joint runs may proceed on any stats (2026-09, v4 ladder).
            if config.training.isotropy_loss_enabled and (stats_mean_max > 0.1 or stats_std_dev > 0.1):
                raise ValueError(
                    "training.joint_encoder requires ~identity latent stats "
                    f"(max|mean| {stats_mean_max:.3f}, max|std-1| {stats_std_dev:.3f}): the isotropy "
                    "penalty pins the RAW latent space while the flow model trains in the "
                    "STANDARDIZED one; those are only the same space near identity stats."
                )
        maybe_modules = [
            getattr(autoencoder, name, None)
            for name in ("encoder", "encoder_connector", "encoder_compression")
        ]
        encoder_modules = [module for module in maybe_modules if module is not None]
        if not encoder_modules:
            raise ValueError(
                "training.joint_encoder requires an autoencoder with encoder/encoder_connector submodules."
            )
        for module in encoder_modules:
            module.requires_grad_(True)
        encoder_parameters = [p for module in encoder_modules for p in module.parameters()]
        if not config.training.isotropy_loss_enabled:
            print(
                "WARNING: joint_encoder without isotropy_loss_enabled -- nothing guards the "
                "latent spectrum against collapse toward trivially-predictable latents.",
                flush=True,
            )

    if model is None:
        model = build_diffusion_backbone(config.diffusion)
    model = model.to(device=device, dtype=dtype)
    model.train()

    optimizer = build_optimizer(model, config)

    start_step = 1
    resume_step = 0
    joint_resume_optimizer_state: dict[str, Any] | None = None
    if config.training.resume_from_checkpoint:
        checkpoint = load_checkpoint(
            config.training.resume_from_checkpoint,
            model=model,
            optimizer=optimizer if config.training.resume_optimizer else None,
            map_location=device,
        )
        resume_step = int(checkpoint["step"])
        start_step = resume_step + 1
        if config.training.max_steps <= resume_step:
            raise ValueError(
                f"max_steps ({config.training.max_steps}) must be greater than the resumed "
                f"checkpoint step ({resume_step})."
            )
        checkpoint_extra = checkpoint.get("extra") or {}
        if bool(checkpoint_extra.get("joint_encoder", False)):
            # A joint checkpoint's diffusion weights are meaningless against any other
            # encoder state: refusing here beats silently decoding from the wrong space.
            if not joint:
                raise ValueError(
                    f"{config.training.resume_from_checkpoint} was written by a joint_encoder run; "
                    "resuming it requires training.joint_encoder: true."
                )
            autoencoder.load_state_dict(checkpoint_extra["autoencoder_state"])
            if config.training.resume_optimizer:
                joint_resume_optimizer_state = checkpoint_extra.get("encoder_optimizer")
        # A plain checkpoint under joint mode is the intended pilot start: the encoder
        # begins at autoencoder.checkpoint and starts drifting from there.

    # Warm-start (new architecture arm from an old checkpoint): load MODEL WEIGHTS ONLY, non-strict,
    # with the fresh optimizer/warmup already built above. Trains from step 1 unless init_start_step
    # relabels the counter (cosmetic: for wandb overlay / cadence alignment; data is unaffected).
    # Only the zero-initialized additions (e.g. the context channel) are permitted to be missing.
    if config.training.init_from_checkpoint:
        if config.training.resume_from_checkpoint:
            raise ValueError("init_from_checkpoint and resume_from_checkpoint are mutually exclusive.")
        allow_missing = getattr(model, "warmstart_allowed_missing", lambda: set())()
        init_checkpoint = load_checkpoint(
            config.training.init_from_checkpoint,
            model=model,
            optimizer=None,
            map_location=device,
            strict=False,
            allow_missing=allow_missing,
        )
        init_extra = init_checkpoint.get("extra") or {}
        if bool(init_extra.get("joint_encoder", False)):
            if not joint:
                raise ValueError(
                    f"{config.training.init_from_checkpoint} was written by a joint_encoder run; "
                    "warm-starting from it requires training.joint_encoder: true."
                )
            autoencoder.load_state_dict(init_extra["autoencoder_state"])
        if config.training.init_start_step is not None:
            if config.training.init_start_step < 0:
                raise ValueError(f"init_start_step must be >= 0, got {config.training.init_start_step}.")
            start_step = config.training.init_start_step + 1
            if config.training.max_steps < start_step:
                raise ValueError(
                    f"max_steps ({config.training.max_steps}) must be >= init_start_step + 1 "
                    f"({start_step}); max_steps is absolute, so set it to init_start_step + "
                    "<additional steps>."
                )

    # Constructed AFTER every autoencoder state mutation above: MasterWeightAdamW snapshots
    # fp32 masters at construction, and step() writes the masters back into the model, so
    # stale masters would silently revert a resumed encoder on the first step. When an
    # encoder-optimizer state is restored, its masters are authoritative (same guarantee,
    # enforced inside load_state_dict).
    enc_optimizer: MasterWeightAdamW | None = None
    if joint:
        enc_optimizer = MasterWeightAdamW(
            parameter_groups(torch.nn.ModuleList(encoder_modules), config.training.weight_decay),
            lr=config.training.encoder_lr,
            betas=config.training.betas,
        )
        if joint_resume_optimizer_state is not None:
            enc_optimizer.load_state_dict(joint_resume_optimizer_state)
        encoder_trainable = sum(p.numel() for p in encoder_parameters)
        print(
            f"joint_encoder: {len(encoder_modules)} encoder modules, {encoder_trainable:,} trainable "
            f"parameters, encoder_lr {config.training.encoder_lr:g}, recon weight "
            f"{config.training.encoder_recon_weight:g}, isotropy "
            f"{'lambda %g' % config.training.isotropy_loss_lambda if config.training.isotropy_loss_enabled else 'OFF'}",
            flush=True,
        )

    trainable_parameters = count_trainable_parameters(model)
    if train_loader is None:
        train_loader, data_info = build_training_loader(config)
    else:
        data_info = {"loaded_slices": [], "probabilities": [], "errors": {}}
    if validation_loaders is None:
        validation_loaders = build_validation_loaders(config)

    # Drift needle: one fixed batch encoded now (eval-mode, deterministic) and re-encoded
    # at every log cadence -- "how fast is the ground moving under the flow model", in
    # standardized-latent RMS units directly comparable to sampler-error sigmas.
    drift_probe_batch: dict[str, Any] | None = None
    drift_reference: torch.Tensor | None = None
    drift_mask: torch.Tensor | None = None
    if joint and validation_loaders:
        first_loader = next(iter(validation_loaders.values()))
        drift_probe_batch = move_batch_to_device(next(iter(first_loader)), device)
        reference = encode_latent_batch(
            autoencoder,
            drift_probe_batch,
            max_words=config.diffusion.max_words,
            latent_dim=config.diffusion.latent_dim,
            dtype=dtype,
            device=device,
            latent_stats=latent_stats,
        )
        drift_reference = reference["z"].clone()
        drift_mask = reference["mask"].clone()

    logger = create_logger(config.training, config=dataclasses.asdict(config))
    logger_url = getattr(getattr(logger, "run", None), "url", None)
    result = DiffusionTrainResult(final_step=0, trainable_parameters=trainable_parameters, logger_url=logger_url)
    train_batches = cycle_batches(train_loader)
    checkpoint_dir = Path(config.training.checkpoint_dir)

    # Generative perplexity (src/perplexity.py). Resolved once here -- the evaluator LM,
    # the length predictor, the prompt list -- so a cadence hit costs generation +
    # scoring only, and a bad evaluator_device fails at step 0 rather than hours in.
    # Imported lazily: src.perplexity reaches back into this module via src.sampling.
    perplexity_every = 0
    perplexity_context: dict[str, Any] | None = None
    if config.perplexity.enabled:
        from src.generation import resolve_length_predictor
        from src.perplexity import describe_spec, load_evaluator, load_prompts

        prompts = None
        if config.perplexity.mode == "prompt":
            if not config.perplexity.prompt_file:
                raise ValueError("perplexity.mode='prompt' requires perplexity.prompt_file.")
            prompts = load_prompts(config.perplexity.prompt_file)
        resolved_predictor = resolve_length_predictor(
            config, device, decode_mode=config.perplexity.decode_mode
        )
        perplexity_context = {
            "evaluator": load_evaluator(
                config.perplexity.evaluator_model,
                config.perplexity.evaluator_device or config.training.device,
                local_files_only=config.perplexity.evaluator_local_files_only,
            ),
            "prompts": prompts,
            "length_predictor": resolved_predictor.predictor,
            "predictor_space": resolved_predictor.space,
        }
        perplexity_every = config.perplexity.every_steps
        print(describe_spec(config, prompts), flush=True)

    try:
        logger.log({"params/diffusion_trainable": float(trainable_parameters)}, step=0)

        if perplexity_context is not None and config.perplexity.baseline_at_start:
            # Reference line: real held-out text scored by the same scorer. The autoencoder
            # is frozen, so this is a run constant -- one point at step 0 is enough. The
            # byte budget matches what the generator produces, because PPL is
            # length-sensitive and a full 2048-byte document is not a fair comparison.
            from src.perplexity import score_baseline_texts

            baseline_bytes = (
                config.perplexity.num_bytes
                or (config.perplexity.num_words or config.diffusion.max_words)
                * config.perplexity.bytes_per_word
            )
            baseline_texts: list[str] = []
            for loader in (validation_loaders or {}).values():
                for batch in loader:
                    baseline_texts.extend(batch["texts"])
                    if len(baseline_texts) >= config.perplexity.baseline_num_texts:
                        break
                if len(baseline_texts) >= config.perplexity.baseline_num_texts:
                    break
            if baseline_texts:
                baseline = score_baseline_texts(
                    perplexity_context["evaluator"],
                    baseline_texts[: config.perplexity.baseline_num_texts],
                    max_bytes=baseline_bytes,
                    max_tokens=config.perplexity.max_eval_tokens,
                )
                logger.log({f"perplexity/{k}": v for k, v in baseline.items()}, step=0)
                print(f"perplexity baseline: {baseline}", flush=True)

        # Actual Training
        for step in range(start_step, config.training.max_steps + 1):
            model.train()
            autoencoder.eval()
            if joint:
                # Decoder stays eval (frozen teacher); only the encoder side trains.
                for module in encoder_modules:
                    module.train()

            schedule_step = step
            schedule_max_steps = config.training.max_steps
            if config.training.reset_lr_schedule_on_resume and resume_step > 0:
                schedule_step = step - resume_step
                schedule_max_steps = config.training.max_steps - resume_step
            step_lr = learning_rate_for_step(schedule_step, config.training, max_steps=schedule_max_steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr
            encoder_step_lr = 0.0
            if enc_optimizer is not None:
                # Same schedule shape as the main LR, scaled to the encoder's peak.
                encoder_step_lr = step_lr * (config.training.encoder_lr / config.training.lr)
                for group in enc_optimizer.param_groups:
                    group["lr"] = encoder_step_lr

            optimizer.zero_grad(set_to_none=True)
            if enc_optimizer is not None:
                enc_optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_words = 0
            step_t_mean = 0.0
            step_z_norm = 0.0
            step_noise_norm = 0.0
            step_mse = 0.0
            step_ce = 0.0
            step_ce_clip_frac = 0.0
            step_prefix_frac = 0.0
            step_seq_lens: list[int] = []
            step_row_logsnr: list[float] = []
            step_row_mse: list[float] = []
            step_prefix_lens: list[int] = []
            step_conditioned_rows = 0
            step_total_rows = 0
            step_recon_ce = 0.0
            step_iso = 0.0
            step_iso_n = 0.0
            step_iso_sum: torch.Tensor | None = None
            step_iso_outer: torch.Tensor | None = None
            started_at = time.perf_counter()

            for _ in range(config.training.gradient_accumulation_steps):
                batch = move_batch_to_device(next(train_batches), device)
                encode = encode_latent_batch_trainable if joint else encode_latent_batch
                latent_batch = encode(
                    autoencoder,
                    batch,
                    max_words=config.diffusion.max_words,
                    latent_dim=config.diffusion.latent_dim,
                    dtype=dtype,
                    device=device,
                    latent_stats=latent_stats,
                )
                ce_context = None
                if config.diffusion.ce_anchor_enabled or joint:
                    ce_context = {
                        "byte_ids": batch["byte_ids"],
                        "labels": batch["labels"],
                        "word_boundaries": batch["word_boundaries"],
                        "lengths": latent_batch["lengths"],
                    }
                loss, metrics = flow_matching_loss(
                    model,
                    latent_batch["z"],
                    latent_batch["mask"],
                    autoencoder=autoencoder,
                    ce_context=ce_context,
                    latent_stats=latent_stats,
                )
                if joint:
                    # Reconstruction CE at the TRUE latents = decoder_ce_anchor evaluated on
                    # the encoder's own output instead of x̂₀: same packing, same truncation
                    # semantics, gradients into z and from there into the encoder. This is
                    # the decodability leash; x0_clamp=0 disables the anchor-only clamp.
                    recon_ce = decoder_ce_anchor(
                        autoencoder,
                        latent_batch["z"],
                        ce_context,
                        latent_stats,
                        x0_clamp=0.0,
                        dtype=module_dtype(autoencoder, latent_batch["z"].dtype),
                    )
                    loss = loss + config.training.encoder_recon_weight * recon_ce
                    step_recon_ce += float(recon_ce.item())
                    if config.training.isotropy_loss_enabled:
                        iso_loss, iso_stats = latent_isotropy_loss(
                            latent_batch["z_raw"],
                            max_latents=config.training.isotropy_loss_max_latents,
                            skip_positions=config.training.isotropy_loss_skip_positions,
                        )
                        loss = loss + config.training.isotropy_loss_lambda * iso_loss
                        step_iso += float(iso_loss.item())
                        if iso_stats:
                            # Pool sufficient statistics across the optimizer step; judge the
                            # spectrum on the pooled covariance, never per micro-batch.
                            step_iso_n += float(iso_stats["n"].item())
                            if step_iso_sum is None:
                                step_iso_sum = iso_stats["sum"].clone()
                                step_iso_outer = iso_stats["sum_outer"].clone()
                            else:
                                step_iso_sum += iso_stats["sum"]
                                step_iso_outer += iso_stats["sum_outer"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite diffusion loss at step {step}: {loss.item()}")

                (loss / config.training.gradient_accumulation_steps).backward()
                words = int(latent_batch["mask"].sum().item())
                step_loss += float(loss.item()) * words
                step_words += words
                step_t_mean += metrics["t_mean"]
                step_z_norm += metrics["z_norm"]
                step_noise_norm += metrics["noise_norm"]
                step_mse += metrics["mse"] * words
                step_ce += metrics["ce"] * words
                step_ce_clip_frac += metrics["ce_clip_frac"]
                step_prefix_frac += metrics["prefix_frac"]
                step_seq_lens.extend(metrics["seq_lens"])
                step_row_logsnr.extend(metrics["row_logsnr"])
                step_row_mse.extend(metrics["row_mse"])
                step_prefix_lens.extend(metrics.get("prefix_lens", []))
                step_conditioned_rows += metrics.get("num_conditioned", 0)
                step_total_rows += metrics["num_rows"]

            grad_norm = clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            encoder_grad_norm = 0.0
            if enc_optimizer is not None:
                # Clip BEFORE step: MasterWeightAdamW copies grads into the fp32 masters
                # inside step(), so the masters see the clipped values.
                encoder_clip = clip_grad_norm_(encoder_parameters, config.training.grad_clip_norm)
                encoder_grad_norm = float(encoder_clip.item() if hasattr(encoder_clip, "item") else encoder_clip)
                enc_optimizer.step()
            optimizer.step()
            elapsed = max(1e-9, time.perf_counter() - started_at)
            result.final_step = step

            if step == 1 or step % config.training.log_every_steps == 0:
                denom = max(1, config.training.gradient_accumulation_steps)
                # Bins with no sampled rows this step are skipped, not logged as 0/NaN.
                mse_by_logsnr = {}
                for lo, hi, name in LOGSNR_BINS:
                    values = [m for m, l in zip(step_row_mse, step_row_logsnr) if lo <= l < hi]
                    if values:
                        mse_by_logsnr[f"train/mse_by_logsnr/{name}"] = sum(values) / len(values)
                joint_metrics: dict[str, Any] = {}
                if joint:
                    joint_metrics = {
                        "train/recon_ce": step_recon_ce / max(1, config.training.gradient_accumulation_steps),
                        "train/isotropy": step_iso / max(1, config.training.gradient_accumulation_steps),
                        "train/encoder_lr": encoder_step_lr,
                        "train/encoder_grad_norm": encoder_grad_norm,
                    }
                    # Collapse gate: pooled-covariance eigenvalue extremes over this step's
                    # latents (needs N >> d for a sound estimate; skip degenerate steps).
                    if step_iso_sum is not None and step_iso_n >= 64:
                        pooled_mean = step_iso_sum / step_iso_n
                        pooled_cov = (
                            step_iso_outer - step_iso_n * torch.outer(pooled_mean, pooled_mean)
                        ) / (step_iso_n - 1)
                        eigenvalues = torch.linalg.eigvalsh(pooled_cov.float())
                        joint_metrics["iso/eig_min"] = float(eigenvalues[0].item())
                        joint_metrics["iso/eig_max"] = float(eigenvalues[-1].item())
                    if drift_reference is not None:
                        drift_now = encode_latent_batch(
                            autoencoder,
                            drift_probe_batch,
                            max_words=config.diffusion.max_words,
                            latent_dim=config.diffusion.latent_dim,
                            dtype=dtype,
                            device=device,
                            latent_stats=latent_stats,
                        )["z"]
                        joint_metrics["joint/latent_drift"] = float(
                            (drift_now - drift_reference)[drift_mask].float().pow(2).mean().sqrt().item()
                        )
                logger.log(
                    {
                        "train/loss": step_loss / max(1, step_words),
                        "train/mse": step_mse / max(1, step_words),
                        "train/ce": step_ce / max(1, step_words),
                        "train/ce_clip_frac": step_ce_clip_frac / denom,
                        "train/lr": step_lr,
                        "train/schedule_step": float(schedule_step),
                        "train/global_step": float(step),
                        "train/grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                        "train/words_per_second": step_words / elapsed,
                        "train/t_mean": step_t_mean / denom,
                        "train/z_norm": step_z_norm / denom,
                        "train/noise_norm": step_noise_norm / denom,
                        "train/prefix_frac": step_prefix_frac / denom,
                        **(
                            {"train/logsnr_mean": sum(step_row_logsnr) / len(step_row_logsnr)}
                            if step_row_logsnr
                            else {}
                        ),
                        **mse_by_logsnr,
                        **joint_metrics,
                        # Distributions (rendered as histograms by wandb/tensorboard):
                        # valid words per training sequence this optimizer step, and the
                        # log-SNR levels the step actually trained on.
                        "data/seq_len": step_seq_lens,
                        **({"train/logsnr": step_row_logsnr} if step_row_logsnr else {}),
                        **(
                            {
                                "cond/conditioned_frac": step_conditioned_rows / max(1, step_total_rows),
                                **({"cond/context_len": step_prefix_lens} if step_prefix_lens else {}),
                            }
                            if config.diffusion.prefix_cond_enabled
                            else {}
                        ),
                    },
                    step=step,
                )

            if validation_loaders and (step % config.training.validate_every_steps == 0 or step == config.training.max_steps):
                validation_losses, health = evaluate_per_slice(
                    model, autoencoder, validation_loaders, config, device, dtype, latent_stats=latent_stats
                )
                result.validation_losses = validation_losses
                logger.log({f"validation/{name}_flow_mse": value for name, value in validation_losses.items()}, step=step)
                logger.log({f"health/{name}": value for name, value in health.items()}, step=step)

            if perplexity_every and (step % perplexity_every == 0 or step == config.training.max_steps):
                from src.perplexity import run_perplexity_eval

                # Every argument is already in memory: the model being trained in its
                # current state, the autoencoder frozen at the top of train_diffusion, and
                # the latent stats loaded once. Nothing is read from disk here.
                #
                # Failures are logged and skipped, not fatal: an evaluator OOM or a bad
                # decode is a lost data point on a side metric, and killing a multi-day
                # run over one is the worse outcome. Setup errors (bad device, missing
                # prompt file) still fail hard, above, before training starts.
                try:
                    perplexity_metrics, perplexity_samples, sample_scores = run_perplexity_eval(
                        model,
                        autoencoder,
                        config,
                        device,
                        dtype,
                        latent_stats=latent_stats,
                        evaluator=perplexity_context["evaluator"],
                        length_predictor=perplexity_context["length_predictor"],
                        predictor_space=perplexity_context["predictor_space"],
                        prompts=perplexity_context["prompts"],
                        return_samples=True,
                    )
                    logger.log(
                        {f"perplexity/{name}": value for name, value in perplexity_metrics.items()}, step=step
                    )
                    # In prompt mode "text" is the continuation only -- the prompt has its
                    # own column and is excluded from the score, so repeating it would
                    # just pad the cell the eye has to scan.
                    logger.log_table(
                        "perplexity/samples",
                        columns=["step", "prompt", "text", "ppl", "nll", "num_tokens"],
                        rows=[
                            [
                                step,
                                sample.prompt or "",
                                sample.continuation if sample.prompt is not None else sample.text,
                                score["ppl"],
                                score["nll"],
                                score["num_tokens"],
                            ]
                            for sample, score in zip(perplexity_samples, sample_scores)
                        ],
                        step=step,
                    )
                except Exception as error:  # noqa: BLE001 -- see comment above
                    print(f"WARNING: perplexity eval failed at step {step}: {error!r}", flush=True)

            if step % config.training.checkpoint_every_steps == 0 or step == config.training.max_steps:
                checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    extra={
                        "config": dataclasses.asdict(config),
                        "autoencoder_checkpoint": config.autoencoder.checkpoint,
                        "trainable_parameters": trainable_parameters,
                        "data": data_info,
                        "validation_losses": result.validation_losses,
                        # Joint runs: the checkpoint is self-contained -- the drifted AE state
                        # (the space these diffusion weights live in) plus the encoder
                        # optimizer with its fp32 masters travel with every checkpoint.
                        **(
                            {
                                "joint_encoder": True,
                                "autoencoder_state": autoencoder.state_dict(),
                                "encoder_optimizer": enc_optimizer.state_dict(),
                            }
                            if joint
                            else {}
                        ),
                        "latent_stats": (
                            {
                                "mean": latent_stats["mean"].cpu(),
                                "std": latent_stats["std"].cpu(),
                                "path": config.autoencoder.latent_stats_path,
                            }
                            if latent_stats is not None
                            else None
                        ),
                    },
                )
                result.checkpoint_paths.append(str(checkpoint_path))
    finally:
        logger.close()

    return result
