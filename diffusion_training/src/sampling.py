import codecs
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from src.diffusion_train import (
    load_latent_stats,
    path_coefficients,
    velocity_to_x0,
    x0_to_velocity,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# A solver advances z from t=0 to t=1: solver(velocity_fn, z, steps, pre_step) -> z.
# ``velocity_fn(z, t_value)`` returns the velocity at scalar time ``t_value``;
# ``pre_step(z, t_value)`` (optional) edits z before each evaluation (used for inpainting).
Solver = Callable[..., torch.Tensor]
_SOLVERS: dict[str, Solver] = {}


def register_solver(name: str) -> Callable[[Solver], Solver]:
    def wrap(fn: Solver) -> Solver:
        _SOLVERS[name] = fn
        return fn

    return wrap


def available_solvers() -> list[str]:
    return sorted(_SOLVERS)


def get_solver(name: str) -> Solver:
    if name not in _SOLVERS:
        raise ValueError(f"Unknown solver {name!r}; available: {available_solvers()}")
    return _SOLVERS[name]


@register_solver("euler")
def _euler_solver(velocity_fn, z, steps, pre_step=None):
    dt = 1.0 / steps
    for index in range(steps):
        t_value = (index + 0.5) / steps
        if pre_step is not None:
            z = pre_step(z, t_value)
        z = z + dt * velocity_fn(z, t_value)
    return z


@register_solver("ab2")
def _ab2_solver(velocity_fn, z, steps, pre_step=None):
    # Two-step Adams-Bashforth: z += dt * (1.5*v_k - 0.5*v_{k-1}). The reused previous
    # velocity linearly extrapolates v across the interval, giving second-order accuracy
    # at the same one-evaluation-per-step cost as "euler". Evaluations sit at interval
    # starts t_k = k/steps (the grid the AB coefficients are derived for), not the
    # midpoints "euler" uses; step 0 has no history and takes a plain Euler step.
    dt = 1.0 / steps
    v_prev = None
    for index in range(steps):
        t_value = index / steps
        if pre_step is not None:
            z = pre_step(z, t_value)
        v = velocity_fn(z, t_value)
        z = z + dt * (v if v_prev is None else 1.5 * v - 0.5 * v_prev)
        v_prev = v
    return z


# Task 3 (Phase 0): step-respacing quadrature. warp(u) = s*u / (1 + (s-1)*u) maps [0,1]->[0,1]
# with warp(0)=0, warp(1)=1; s < 1 pushes the grid toward low t so more steps land in the
# informative noise band t in [0.06, 0.35]. s = 0.333 gives ~46% of steps in the band (vs
# ~29% uniform). The velocity field is untouched -- this is pure quadrature. At s = 1.0 the
# warp is the identity and the integration is bit-identical to "euler" (gate below).
_EULER_WARP_S = 0.333


def _time_warp(u: float, s: float) -> float:
    return s * u / (1.0 + (s - 1.0) * u)


def _euler_warped_core(velocity_fn, z, steps, pre_step, s):
    for index in range(steps):
        t_lo = _time_warp(index / steps, s)
        t_hi = _time_warp((index + 1) / steps, s)
        dt = t_hi - t_lo
        t_value = 0.5 * (t_lo + t_hi)
        if pre_step is not None:
            z = pre_step(z, t_value)
        z = z + dt * velocity_fn(z, t_value)
    return z


@register_solver("euler_warped")
def _euler_warped_solver(velocity_fn, z, steps, pre_step=None):
    return _euler_warped_core(velocity_fn, z, steps, pre_step, _EULER_WARP_S)


def _make_velocity_fn(
    model,
    mask: torch.Tensor,
    dtype: torch.dtype,
    clean_mask: torch.Tensor | None = None,
    z_clean: torch.Tensor | None = None,
    x0_scale: float = 1.0,
    z_context: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
):

    self_cond = getattr(model.config, "self_conditioning", False)
    path = getattr(model.config, "path", "linear")
    timeshift = getattr(model.config, "path_timeshift", 1.0)
    prediction_mode = getattr(model.config, "prediction", "velocity")
    # Flag-mode checkpoints receive the learned clean-position marker in the model
    # call itself; clean_mask arrives [B, W, 1] here, the model expects [B, W].
    flag_mode = (
        getattr(model.config, "prefix_cond_enabled", False)
        and getattr(model.config, "prefix_cond_mode", "data") == "flag"
    )
    channel_mode = (
        getattr(model.config, "prefix_cond_enabled", False)
        and getattr(model.config, "prefix_cond_mode", "data") == "channel"
    )
    flag_kwargs = {"clean_mask": clean_mask.squeeze(-1)} if (flag_mode and clean_mask is not None) else {}
    ctx_kwargs = (
        {"context": z_context, "context_mask": context_mask.squeeze(-1)}
        if (channel_mode and z_context is not None and context_mask is not None)
        else {}
    )
    do_cfg = bool(ctx_kwargs) and cfg_scale != 1.0
    state: dict[str, torch.Tensor | None] = {"z_self": None}

    def velocity_fn(z: torch.Tensor, t_value: float) -> torch.Tensor:
        t = torch.full((z.size(0),), t_value, device=z.device)
        raw_cond = model(z, t, mask, state["z_self"], **flag_kwargs, **ctx_kwargs)
        if do_cfg:  # null-context forward, then extrapolate the drive (feedback stays conditional)
            raw_uncond = model(z, t, mask, state["z_self"], **flag_kwargs)
            raw_drive = raw_uncond + cfg_scale * (raw_cond - raw_uncond)
        else:
            raw_drive = raw_cond
        if prediction_mode == "x0":
            x0_drive = raw_drive * x0_scale if x0_scale != 1.0 else raw_drive
            velocity = x0_to_velocity(z, x0_drive, t, path, timeshift)
            x0_self = raw_cond
        else:
            velocity = raw_drive
            x0_self = velocity_to_x0(z, raw_cond, t, path, timeshift) if self_cond else None
        if self_cond:
            if clean_mask is not None and z_clean is not None:
                x0_self = torch.where(clean_mask, z_clean, x0_self)
            state["z_self"] = x0_self.to(dtype)
        return velocity

    return velocity_fn


@torch.no_grad()
def sample_latents(
    model,
    num_samples: int,
    num_words: int,
    latent_dim: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    solver: str = "euler",
    x0_scale: float = 1.0,
) -> torch.Tensor:
    # Path-agnostic: Euler-integrating the learned velocity from z0 ~ N(0, I) transports
    # noise -> data for any interpolation path with sigma(0)=1 (linear and cosine both
    # satisfy this), so no path coefficients are needed here.
    z = torch.randn((num_samples, num_words, latent_dim), device=device, dtype=dtype)
    mask = torch.ones((num_samples, num_words), device=device, dtype=torch.bool)
    velocity_fn = _make_velocity_fn(model, mask, dtype, x0_scale=x0_scale)
    return get_solver(solver)(velocity_fn, z, steps)


@torch.no_grad()
def sample_latents_conditional(
    model,
    z_known: torch.Tensor,
    known_mask: torch.Tensor,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    solver: str = "euler",
    conditioning: str = "interpolant",
    cfg_scale: float = 1.0,
) -> torch.Tensor:

    if conditioning not in ("interpolant", "clean", "channel"):
        raise ValueError(f"Unknown conditioning mode {conditioning!r}; use 'interpolant', 'clean', or 'channel'.")
    z_known = z_known.to(device=device, dtype=dtype)
    known = known_mask.to(device=device, dtype=torch.bool).unsqueeze(-1)
    num_samples, num_words, _ = z_known.shape
    # eps is drawn in all modes (unused by "clean") so equal seeds give equal starting
    # noise across modes — useful for clean-vs-interpolant comparisons.
    eps = torch.randn_like(z_known)
    z = torch.randn_like(z_known)
    path = getattr(model.config, "path", "linear")
    timeshift = getattr(model.config, "path_timeshift", 1.0)
    attn_mask = torch.ones((num_samples, num_words), device=device, dtype=torch.bool)

    if conditioning == "clean":
        velocity_fn = _make_velocity_fn(model, attn_mask, dtype, clean_mask=known, z_clean=z_known)

        def pre_step(z: torch.Tensor, t_value: float) -> torch.Tensor:
            return torch.where(known, z_known, z)

    elif conditioning == "channel":
        # Context on its own channel; z_t at known positions still follows the interpolant, and the
        # self-conditioning feedback is clamped to the truth there (clean_mask/z_clean), matching
        # channel-mode training. cfg_scale drives classifier-free guidance on context strength.
        velocity_fn = _make_velocity_fn(
            model,
            attn_mask,
            dtype,
            clean_mask=known,
            z_clean=z_known,
            z_context=z_known,
            context_mask=known,
            cfg_scale=cfg_scale,
        )

        def pre_step(z: torch.Tensor, t_value: float) -> torch.Tensor:
            alpha, sigma, _, _ = path_coefficients(t_value, path, timeshift)
            return torch.where(known, alpha * z_known + sigma * eps, z)

    else:
        velocity_fn = _make_velocity_fn(model, attn_mask, dtype)

        def pre_step(z: torch.Tensor, t_value: float) -> torch.Tensor:
            alpha, sigma, _, _ = path_coefficients(t_value, path, timeshift)
            return torch.where(known, alpha * z_known + sigma * eps, z)

    z = get_solver(solver)(velocity_fn, z, steps, pre_step)
    return torch.where(known, z_known, z)


def _clip_boundaries(boundaries: torch.Tensor, length: int) -> torch.Tensor:
    clipped = boundaries[boundaries < length]
    if clipped.numel() == 0:
        clipped = boundaries[:1]
    if int(clipped[-1].item()) != length:
        clipped = torch.cat([clipped, clipped.new_tensor([length])])
    return clipped


@torch.no_grad()
def generate_bytes(
    autoencoder: torch.nn.Module,
    z_words: torch.Tensor,
    num_bytes: int,
    bytes_per_word: int,
    seed_byte: int,
    device: torch.device,
    word_lengths: list[int] | None = None,
    prompt_byte_ids: list[int] | None = None,
) -> list[int]:


    if word_lengths is not None:
        cumulative = [0]
        for length in word_lengths[: z_words.size(1)]:
            cumulative.append(cumulative[-1] + max(1, int(length)))
        full_boundaries = torch.tensor(cumulative, device=device, dtype=torch.int32)
    else:
        full_boundaries = torch.arange(
            0,
            z_words.size(1) * bytes_per_word + 1,
            bytes_per_word,
            device=device,
            dtype=torch.int32,
        )
    if prompt_byte_ids:
        generated = [int(b) % 256 for b in prompt_byte_ids]
    else:
        generated = [int(seed_byte) % 256]
    while len(generated) < num_bytes:
        current_length = len(generated)
        byte_ids = torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)
        boundaries = _clip_boundaries(full_boundaries, current_length)
        word_count = min(boundaries.numel() - 1, z_words.size(1))
        boundaries = boundaries[: word_count + 1]
        logits = autoencoder.decode(byte_ids, [z_words[:, :word_count, :]], [boundaries])
        generated.append(int(logits[0, current_length - 1].argmax(dim=-1).item()))
    return generated


@torch.no_grad()
def decode_latents(
    autoencoder: torch.nn.Module,
    z_words: torch.Tensor,
    num_bytes: int,
    bytes_per_word: int,
    seed_byte: int,
    device: torch.device,
    word_lengths: list[int] | None = None,
    prompt_byte_ids: list[int] | None = None,
) -> str:
    generated = generate_bytes(
        autoencoder,
        z_words,
        num_bytes=num_bytes,
        bytes_per_word=bytes_per_word,
        seed_byte=seed_byte,
        device=device,
        word_lengths=word_lengths,
        prompt_byte_ids=prompt_byte_ids,
    )
    return bytes(generated).decode("utf-8", errors="replace")


def _splitter_boundaries(byte_ids: list[int], splitter) -> list[int]:

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(bytes(byte_ids), final=False)
    words = splitter.encode(text)
    cumulative = [0]
    position = 0
    for word in words[:-1]:
        if byte_ids[position : position + len(word)] != word:
            break
        position += len(word)
        cumulative.append(position)
    cumulative.append(len(byte_ids))
    return cumulative


@torch.no_grad()
def generate_bytes_splitter(
    autoencoder: torch.nn.Module,
    z_words: torch.Tensor,
    seed_byte: int,
    device: torch.device,
    max_bytes: int,
    prompt_byte_ids: list[int] | None = None,
) -> list[int]:

    from src.data import TextCollator  # lazy: pulls in HATSplitter

    # Only the splitter (with training defaults, e.g. max_word_size) is used.
    splitter = TextCollator(sequence_length=max_bytes).splitter
    num_words = z_words.size(1)
    if prompt_byte_ids:
        generated = [int(b) % 256 for b in prompt_byte_ids]
    else:
        generated = [int(seed_byte) % 256]
    while len(generated) < max_bytes:
        cumulative = _splitter_boundaries(generated, splitter)
        if len(cumulative) - 1 > num_words:
            break
        word_count = len(cumulative) - 1
        boundaries = torch.tensor(cumulative, device=device, dtype=torch.int32)
        byte_ids = torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)
        logits = autoencoder.decode(byte_ids, [z_words[:, :word_count, :]], [boundaries])
        generated.append(int(logits[0, len(generated) - 1].argmax(dim=-1).item()))
        # Degeneration guard: a dense run of undecodable bytes means the byte sampler has
        # left valid text space entirely (observed 2026-08-08: one sample wedged into
        # ~1.6kB of U+FFFD and padded the canvas to max_bytes). Trim the garbage window
        # and stop; legitimate text never produces 16 replacement chars in 48 bytes (a
        # multibyte char split at the window edge yields at most a couple).
        if len(generated) >= 48:
            tail = bytes(generated[-48:]).decode("utf-8", errors="replace")
            if tail.count("�") >= 16:
                generated = generated[:-48]
                break
    # The last byte may have opened a word past the final latent; trim it off.
    cumulative = _splitter_boundaries(generated, splitter)
    if len(cumulative) - 1 > num_words:
        generated = generated[: cumulative[num_words]]
    return generated


@dataclass
class PromptEncoding:
    """A prompt encoded to word latents, with exact splitter segmentation."""

    text: str  # what actually got encoded (may be truncated at data.sequence_length)
    byte_ids: list[int]  # raw prompt bytes; len == word_lengths sum
    word_lengths: list[int]  # per-word byte lengths from the HAT splitter (exact)
    z_words: torch.Tensor  # [1, num_words, latent_dim], UNstandardized, float32
    truncated: bool


@torch.no_grad()
def encode_prompt(
    autoencoder: torch.nn.Module,
    text: str,
    sequence_length: int,
    device: torch.device,
) -> PromptEncoding:
    """Text -> per-word latents via the frozen autoencoder (splitter segmentation)."""

    from src.data import TextCollator  # lazy: pulls in HATSplitter

    collator = TextCollator(sequence_length=sequence_length)
    batch = collator([text])
    byte_ids = batch["byte_ids"].to(device)
    boundary = batch["word_boundaries"][0].to(device)
    z_words = autoencoder.encode(byte_ids, [boundary])[0].float()
    if z_words.ndim == 2:
        z_words = z_words.unsqueeze(0)
    word_lengths = (boundary[1:] - boundary[:-1]).tolist()
    prompt_bytes = byte_ids[0, : int(boundary[-1].item())].tolist()
    encoded_text = collator.decode_bytes(prompt_bytes)
    return PromptEncoding(
        text=encoded_text,
        byte_ids=prompt_bytes,
        word_lengths=word_lengths,
        z_words=z_words,
        truncated=encoded_text.strip() != text.strip(),
    )


def resolve_latent_stats(checkpoint: dict, config, device: torch.device) -> dict | None:
    """Latent mean/std for (de)standardization: checkpoint extra wins, then config path."""

    latent_stats = checkpoint.get("extra", {}).get("latent_stats")
    if latent_stats is None and config.autoencoder.latent_stats_path:
        latent_stats = load_latent_stats(
            config.autoencoder.latent_stats_path, device, config.diffusion.latent_dim
        )
    if latent_stats is None:
        print(
            "WARNING: no latent stats in checkpoint or config; decoding raw flow outputs. "
            "Checkpoints trained without standardization are known to produce noise-prior latents.",
            file=sys.stderr,
        )
        return None
    return {
        "mean": latent_stats["mean"].to(device=device, dtype=torch.float32),
        "std": latent_stats["std"].to(device=device, dtype=torch.float32),
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Highest-step ``step_*.pt`` in ``checkpoint_dir`` (resolved against CWD, then repo root)."""

    raw = Path(checkpoint_dir)
    candidates = [raw] if raw.is_absolute() else [raw, _REPO_ROOT / raw]
    directory = next((d for d in candidates if d.is_dir()), None)
    steps: list[tuple[int, Path]] = []
    if directory is not None:
        for path in directory.glob("step_*.pt"):
            try:
                steps.append((int(path.stem.split("_")[1]), path))
            except (IndexError, ValueError):
                continue
    if not steps:
        raise FileNotFoundError(
            f"No step_*.pt checkpoints found for training.checkpoint_dir={str(checkpoint_dir)!r} "
            f"(looked in {[str(c) for c in candidates]}); pass --checkpoint explicitly."
        )
    return max(steps)[1]
