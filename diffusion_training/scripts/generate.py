import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import load_frozen_autoencoder
from src.generation import (
    GenerationSpec,
    generate_texts,
    resolve_conditioning,
    resolve_length_predictor,
    resolve_num_words,
)
from src.sampling import available_solvers, find_latest_checkpoint, resolve_latent_stats
from src.train import apply_tf32, diffusion_dtype_from_config
from src.utils import load_checkpoint, load_diffusion_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Diffusion YAML config (bare names resolve against configs/training/).")
    parser.add_argument(
        "--checkpoint",
        help="Diffusion checkpoint; defaults to the highest step_*.pt in the config's checkpoint_dir.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Context text; encoded and clamped as a known prefix.")
    prompt_group.add_argument("--prompt-file", help="Read the prompt text from a file.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--num-words",
        type=int,
        help="Total word canvas incl. prompt words (default: diffusion.max_words).",
    )
    parser.add_argument("--num-bytes", type=int, help="Total decode length (default: sum of word lengths).")
    parser.add_argument("--bytes-per-word", type=int, default=6)
    parser.add_argument(
        "--decode-mode",
        choices=["lengths", "splitter"],
        default="lengths",
        help="'lengths' (default): byte->latent routing is fixed up front from the length "
        "predictor or the --bytes-per-word grid. 'splitter': HAT-paper-style decode — the "
        "splitter re-segments the sampled bytes after every step, so boundaries are an "
        "output of decoding; no length predictor is used and --num-bytes is only a safety "
        "cap (default: data.sequence_length).",
    )
    parser.add_argument(
        "--length-predictor",
        help="Length-predictor artifact (scripts/train_length_predictor.py); overrides "
        "config.length_predictor.artifact_path. Replaces the fixed bytes-per-word grid.",
    )
    parser.add_argument(
        "--length-predictor-space",
        choices=["raw", "standardized"],
        default=None,
        help="Fallback latent space for space-less (legacy v1) artifacts: 'raw' for "
        "length_predictor-compress256-raw.pt (default), 'standardized' for the legacy 4096-dim artifact. "
        "Ignored when the artifact records its own space (v2).",
    )
    parser.add_argument("--sampling-steps", type=int, help="Default: diffusion.sampling_steps from the config.")
    parser.add_argument("--solver", default="euler", choices=available_solvers())
    parser.add_argument(
        "--x0-scale",
        type=float,
        default=1.0,
        help="Inflate predicted clean-latent deviations at sampling time (x0-prediction only). "
        "1.0 = unchanged; >1 counteracts mean-hedging shrinkage (Phase 0 Task 4a). Unconditional mode only.",
    )
    parser.add_argument(
        "--conditioning",
        choices=["auto", "interpolant", "clean", "channel"],
        default="auto",
        help="Prompt-mode clamp: 'clean' holds prompt latents fully clean each step (matches "
        "prefix-conditioned training), 'interpolant' is replacement inpainting (unconditional "
        "training), 'channel' feeds context on the dedicated context channel (prefix_cond_mode "
        "'channel'). 'auto' picks by config.diffusion.prefix_cond_enabled / prefix_cond_mode.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance on context strength (channel conditioning only). 1.0 = off; "
        ">1 overweights the context. Ignored unless conditioning resolves to 'channel'.",
    )
    parser.add_argument("--seed", type=int, help="RNG seed for reproducible sampling.")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed-byte", type=int, default=32, help="Unconditional decode seed byte (ignored in prompt mode).")
    seed_group.add_argument(
        "--seed-text",
        help="Seed the unconditional decode with these true UTF-8 bytes instead of --seed-byte. "
        "The bytes are teacher-forced and appear verbatim at the start of the output; the "
        "latents themselves stay unconditional (use --prompt to condition them). Ignored in prompt mode.",
    )
    parser.add_argument("--output", help="Also write the generated report to this file.")
    parser.add_argument("--device")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    if args.allow_download:
        config.autoencoder.local_files_only = False
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or config.training.device)
    apply_tf32(config.training)
    # Diffusion dtype: types the model and the ODE. The autoencoder keeps
    # config.training.dtype (load_frozen_autoencoder reads it directly).
    dtype = diffusion_dtype_from_config(config.training)

    checkpoint_path = args.checkpoint
    auto_selected = False
    if checkpoint_path is None:
        checkpoint_path = str(find_latest_checkpoint(config.training.checkpoint_dir))
        auto_selected = True

    prompt_text = None
    if args.prompt is not None:
        prompt_text = args.prompt
    elif args.prompt_file is not None:
        prompt_text = Path(args.prompt_file).read_text().strip()
    if prompt_text is not None and not prompt_text.strip():
        raise SystemExit("Prompt is empty.")
    if prompt_text is None and args.seed_text is not None and not args.seed_text:
        raise SystemExit("--seed-text is empty.")

    autoencoder = load_frozen_autoencoder(config, device)
    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    latent_stats = resolve_latent_stats(checkpoint, config, device)

    resolved = resolve_length_predictor(
        config,
        device,
        decode_mode=args.decode_mode,
        override_path=args.length_predictor,
        override_space=args.length_predictor_space,
        config_name=args.config,
    )

    num_words = resolve_num_words(config, args.num_words)
    steps = args.sampling_steps or config.diffusion.sampling_steps

    report: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report.append(line)

    emit(f"checkpoint={checkpoint_path}{' (auto-selected)' if auto_selected else ''}")
    emit(f"step={checkpoint.get('step')}")
    conditioning = resolve_conditioning(config, args.conditioning)
    emit(
        f"mode={'prompt' if prompt_text is not None else 'unconditional'} num_words={num_words} "
        f"sampling_steps={steps} solver={args.solver} seed={args.seed}"
        + (f" conditioning={conditioning}" if prompt_text is not None else "")
    )
    emit(f"standardized={latent_stats is not None} {resolved.description}")

    if prompt_text is None:
        if args.seed_text is not None:
            emit(
                f"note: decode seeded with {len(args.seed_text.encode('utf-8'))} true byte(s) from --seed-text "
                "(teacher-forced; latents are still unconditional)"
            )
        else:
            emit("note: the first 1-2 words are unreliable (decode is seeded with a space byte)")

    spec = GenerationSpec(
        num_samples=args.num_samples,
        num_words=num_words,
        num_bytes=args.num_bytes,
        bytes_per_word=args.bytes_per_word,
        sampling_steps=steps,
        solver=args.solver,
        decode_mode=args.decode_mode,
        conditioning=conditioning,
        x0_scale=args.x0_scale,
        cfg_scale=args.cfg_scale,
        seed_byte=args.seed_byte,
        seed_text=args.seed_text,
    )
    samples = generate_texts(
        model,
        autoencoder,
        config,
        spec,
        device=device,
        dtype=dtype,
        latent_stats=latent_stats,
        length_predictor=resolved.predictor,
        predictor_space=resolved.space,
        prompt_text=prompt_text,
    )

    if prompt_text is not None and samples:
        emit(f"prompt_words={samples[0].prompt_words} prompt_bytes={samples[0].prompt_bytes}")

    for index, sample in enumerate(samples):
        emit()
        emit(f"SAMPLE {index + 1}")
        if prompt_text is None:
            emit(sample.text)
        else:
            emit(f"PROMPT       | {sample.prompt}")
            emit(f"CONTINUATION | {sample.continuation}")

    if args.output:
        Path(args.output).write_text("\n".join(report) + "\n")
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
