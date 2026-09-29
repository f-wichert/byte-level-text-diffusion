import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from src.sampling import (
    encode_prompt,
    generate_bytes,
    generate_bytes_splitter,
    sample_latents,
    sample_latents_conditional,
)
from src.utils import module_dtype


@dataclass
class GenerationSpec:
    num_samples: int = 1
    num_words: int | None = None  # None -> diffusion.max_words
    num_bytes: int | None = None  # None -> sum of predicted word lengths
    bytes_per_word: int = 6
    sampling_steps: int | None = None  # None -> diffusion.sampling_steps
    solver: str = "euler"
    decode_mode: str = "lengths"  # lengths | splitter
    conditioning: str = "auto"  # auto | interpolant | clean | channel
    x0_scale: float = 1.0
    cfg_scale: float = 1.0  # classifier-free guidance on context strength (channel conditioning only)
    seed_byte: int = 32
    seed_text: str | None = None


@dataclass
class GeneratedSample:
    """One decoded sample. ``prompt`` / ``continuation`` are None in unconditional mode."""

    text: str  # full decoded text (prompt included, in prompt mode)
    prompt: str | None = None
    continuation: str | None = None
    num_bytes: int = 0
    # Prompt-mode only: the splitter's exact segmentation of the prompt. Reported rather
    # than recomputed because the prompt may have been truncated at data.sequence_length.
    prompt_words: int = 0
    prompt_bytes: int = 0


@dataclass
class ResolvedPredictor:
    """A loaded length predictor plus the latent space its inputs must be in."""

    predictor: object | None
    space: str | None
    description: str


def resolve_length_predictor(
    config,
    device: torch.device,
    *,
    decode_mode: str = "lengths",
    override_path: str | None = None,
    override_space: str | None = None,
    config_name: str = "<config>",
) -> ResolvedPredictor:
    if decode_mode == "splitter":
        return ResolvedPredictor(
            None, None, "decode_mode=splitter (boundaries from sampled bytes; length predictor unused)"
        )

    predictor_path = override_path or config.length_predictor.artifact_path
    if predictor_path is None:
        return ResolvedPredictor(None, None, "length_predictor=None (fixed bytes-per-word grid)")

    if override_path is None and not Path(predictor_path).exists():
        raise SystemExit(
            f"config.length_predictor.artifact_path={predictor_path} does not exist; run:\n"
            f"  python diffusion_training/scripts/train_length_predictor.py {config_name}"
        )

    from src.length_predictor import load_length_predictor

    predictor = load_length_predictor(predictor_path, device)
    source = "--length-predictor flag" if override_path else "config.length_predictor.artifact_path"
    # Space: the artifact's recorded value wins; fall back to the flag for space-less v1.
    if predictor.space is not None:
        space = predictor.space
        if override_space is not None and override_space != space:
            print(
                f"WARNING: --length-predictor-space {override_space} conflicts with the "
                f"artifact-recorded space {space!r}; using the recorded value.",
                file=sys.stderr,
            )
    else:
        space = override_space or "raw"
    return ResolvedPredictor(
        predictor, space, f"length_predictor={predictor_path} (from {source}) space={space}"
    )


def resolve_conditioning(config, conditioning: str) -> str:
    """'auto' picks by config.diffusion.prefix_cond_enabled / prefix_cond_mode; others pass through."""

    if conditioning != "auto":
        return conditioning
    if not getattr(config.diffusion, "prefix_cond_enabled", False):
        return "interpolant"
    return "channel" if getattr(config.diffusion, "prefix_cond_mode", "data") == "channel" else "clean"


def resolve_num_words(config, num_words: int | None) -> int:
    max_words = config.diffusion.max_words
    resolved = num_words or max_words
    if resolved > max_words:
        print(
            f"WARNING: num_words {resolved} exceeds diffusion.max_words; clamping to {max_words}.",
            file=sys.stderr,
        )
        resolved = max_words
    return resolved


@torch.no_grad()
def generate_texts(
    model,
    autoencoder,
    config,
    spec: GenerationSpec,
    device: torch.device,
    dtype: torch.dtype,
    latent_stats: dict | None,
    length_predictor=None,
    predictor_space: str | None = None,
    prompt_text: str | None = None,
) -> list[GeneratedSample]:

    num_words = resolve_num_words(config, spec.num_words)
    steps = spec.sampling_steps or config.diffusion.sampling_steps
    conditioning = resolve_conditioning(config, spec.conditioning)
    decode_dtype = module_dtype(autoencoder, dtype)
    samples: list[GeneratedSample] = []

    if prompt_text is None:
        seed_byte_ids = None
        if spec.seed_text is not None:
            if not spec.seed_text:
                raise ValueError("seed_text is empty.")
            seed_byte_ids = list(spec.seed_text.encode("utf-8"))
        z_samples = sample_latents(
            model,
            num_samples=spec.num_samples,
            num_words=num_words,
            latent_dim=config.diffusion.latent_dim,
            steps=steps,
            device=device,
            dtype=dtype,
            solver=spec.solver,
            x0_scale=spec.x0_scale,
        )
        for z_words in z_samples:
            z_std = z_words.unsqueeze(0).float()
            z_raw = z_std
            if latent_stats is not None:
                z_raw = z_std * latent_stats["std"] + latent_stats["mean"]
            word_lengths = None
            if length_predictor is not None:
                z_pred = z_raw if predictor_space == "raw" else z_std
                word_lengths = length_predictor.predict_lengths(z_pred[0]).tolist()
            z_decode = z_raw.to(dtype=decode_dtype)
            if spec.decode_mode == "splitter":
                generated = generate_bytes_splitter(
                    autoencoder,
                    z_decode,
                    seed_byte=spec.seed_byte,
                    device=device,
                    max_bytes=spec.num_bytes or config.data.sequence_length,
                    prompt_byte_ids=seed_byte_ids,
                )
            else:
                if word_lengths is not None:
                    num_bytes = spec.num_bytes or sum(word_lengths)
                else:
                    num_bytes = spec.num_bytes or num_words * spec.bytes_per_word
                generated = generate_bytes(
                    autoencoder,
                    z_decode,
                    num_bytes=num_bytes,
                    bytes_per_word=spec.bytes_per_word,
                    seed_byte=spec.seed_byte,
                    device=device,
                    word_lengths=word_lengths,
                    prompt_byte_ids=seed_byte_ids,
                )
            samples.append(
                GeneratedSample(
                    text=bytes(generated).decode("utf-8", errors="replace"),
                    num_bytes=len(generated),
                )
            )
        return samples

    prompt = encode_prompt(autoencoder, prompt_text, config.data.sequence_length, device)
    if prompt.truncated:
        print(
            f"WARNING: prompt truncated at data.sequence_length={config.data.sequence_length} bytes; "
            f"encoding: {prompt.text!r}",
            file=sys.stderr,
        )
    prompt_words = len(prompt.word_lengths)
    if prompt_words >= num_words:
        raise ValueError(
            f"Prompt has {prompt_words} words but the canvas is only {num_words} "
            f"(diffusion.max_words={config.diffusion.max_words}); shorten the prompt or raise num_words."
        )

    z_prompt = prompt.z_words
    if latent_stats is not None:
        z_prompt = (z_prompt - latent_stats["mean"]) / latent_stats["std"]
    else:
        print(
            "WARNING: prompt conditioning without latent stats is untested; the model's "
            "working space is assumed to be raw latents.",
            file=sys.stderr,
        )
    z_known = torch.zeros(
        (spec.num_samples, num_words, config.diffusion.latent_dim), device=device, dtype=torch.float32
    )
    z_known[:, :prompt_words] = z_prompt
    known_mask = torch.zeros((spec.num_samples, num_words), dtype=torch.bool)
    known_mask[:, :prompt_words] = True

    z_samples = sample_latents_conditional(
        model,
        z_known,
        known_mask,
        steps=steps,
        device=device,
        dtype=dtype,
        solver=spec.solver,
        conditioning=conditioning,
        cfg_scale=spec.cfg_scale,
    )
    for z_words in z_samples:
        z_std = z_words.unsqueeze(0).float()
        z_raw = z_std
        if latent_stats is not None:
            z_raw = z_std * latent_stats["std"] + latent_stats["mean"]
        z_decode = z_raw.to(dtype=decode_dtype)
        if spec.decode_mode == "splitter":
            # Prompt bytes came from the splitter, so re-segmenting reproduces their
            # exact boundaries; the continuation's boundaries emerge byte by byte.
            generated = generate_bytes_splitter(
                autoencoder,
                z_decode,
                seed_byte=spec.seed_byte,
                device=device,
                max_bytes=spec.num_bytes or config.data.sequence_length,
                prompt_byte_ids=prompt.byte_ids,
            )
        else:
            # Prompt boundaries are exact (from the splitter); only the continuation is
            # guessed via the length predictor or the fixed grid.
            if length_predictor is not None:
                z_pred = z_raw if predictor_space == "raw" else z_std
                predicted = length_predictor.predict_lengths(z_pred[0]).tolist()
                word_lengths = list(prompt.word_lengths) + predicted[prompt_words:num_words]
            else:
                word_lengths = list(prompt.word_lengths) + [spec.bytes_per_word] * (num_words - prompt_words)
            num_bytes = spec.num_bytes or sum(word_lengths)
            if num_bytes <= len(prompt.byte_ids):
                print(
                    f"WARNING: num_bytes {num_bytes} does not exceed the prompt length "
                    f"({len(prompt.byte_ids)} bytes); nothing will be generated.",
                    file=sys.stderr,
                )
            generated = generate_bytes(
                autoencoder,
                z_decode,
                num_bytes=num_bytes,
                bytes_per_word=spec.bytes_per_word,
                seed_byte=spec.seed_byte,
                device=device,
                word_lengths=word_lengths,
                prompt_byte_ids=prompt.byte_ids,
            )
        continuation = bytes(generated[len(prompt.byte_ids) :]).decode("utf-8", errors="replace")
        samples.append(
            GeneratedSample(
                text=bytes(generated).decode("utf-8", errors="replace"),
                prompt=prompt.text,
                continuation=continuation,
                num_bytes=len(generated),
                prompt_words=prompt_words,
                prompt_bytes=len(prompt.byte_ids),
            )
        )
    return samples
