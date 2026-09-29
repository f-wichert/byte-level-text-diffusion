import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diffusion_model import build_diffusion_backbone
from src.diffusion_train import load_frozen_autoencoder
from src.generation import resolve_length_predictor
from src.perplexity import (
    describe_spec,
    load_evaluator,
    load_prompts,
    run_perplexity_eval,
    score_baseline_texts,
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
    parser.add_argument("--mode", choices=["prompt", "unconditional"], help="Default: perplexity.mode.")
    parser.add_argument("--prompt-file", help="Default: perplexity.prompt_file.")
    parser.add_argument("--num-samples", type=int, help="Default: perplexity.num_samples.")
    parser.add_argument("--num-words", type=int, help="Default: perplexity.num_words, else diffusion.max_words.")
    parser.add_argument("--sampling-steps", type=int, help="Default: perplexity.sampling_steps, else diffusion.sampling_steps.")
    parser.add_argument("--solver", choices=available_solvers(), help="Default: perplexity.solver.")
    parser.add_argument("--decode-mode", choices=["lengths", "splitter"], help="Default: perplexity.decode_mode.")
    parser.add_argument("--seed", type=int, help="Default: perplexity.seed. Keep fixed for comparable numbers.")
    parser.add_argument("--evaluator-model", help="Default: perplexity.evaluator_model (e.g. gpt2, gpt2-large).")
    parser.add_argument("--evaluator-device", help="Device for the scoring LM. Default: perplexity.evaluator_device.")
    parser.add_argument("--device", help="Device for the diffusion model and autoencoder. Default: training.device.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also score real held-out text as a reference line (needs the validation loaders).",
    )
    parser.add_argument("--print-samples", action="store_true", help="Print each generated sample with the metrics.")
    parser.add_argument("--output", help="Write the metrics as JSON to this file.")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_diffusion_config(args.config)
    if args.allow_download:
        config.autoencoder.local_files_only = False

    # CLI flags override the config block, then everything downstream reads the config --
    # so the resolved spec logged below is the one actually used.
    perplexity = config.perplexity
    for name in ("mode", "prompt_file", "num_samples", "num_words", "sampling_steps", "solver", "decode_mode", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(perplexity, name, value)
    if args.evaluator_model is not None:
        perplexity.evaluator_model = args.evaluator_model
    if args.evaluator_device is not None:
        perplexity.evaluator_device = args.evaluator_device

    device = torch.device(args.device or config.training.device)
    apply_tf32(config.training)
    dtype = diffusion_dtype_from_config(config.training)

    checkpoint_path = args.checkpoint
    auto_selected = False
    if checkpoint_path is None:
        checkpoint_path = str(find_latest_checkpoint(config.training.checkpoint_dir))
        auto_selected = True

    prompts = None
    if perplexity.mode == "prompt":
        if not perplexity.prompt_file:
            raise SystemExit("mode='prompt' needs a prompt file; set perplexity.prompt_file or --prompt-file.")
        prompts = load_prompts(perplexity.prompt_file)

    # Load the evaluator first: a bad --evaluator-device should fail before the several
    # seconds spent materialising the 7B-backed autoencoder.
    evaluator = load_evaluator(
        perplexity.evaluator_model,
        perplexity.evaluator_device or config.training.device,
        local_files_only=perplexity.evaluator_local_files_only,
    )

    autoencoder = load_frozen_autoencoder(config, device)
    model = build_diffusion_backbone(config.diffusion).to(device=device, dtype=dtype)
    checkpoint = load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    latent_stats = resolve_latent_stats(checkpoint, config, device)
    resolved = resolve_length_predictor(
        config,
        device,
        decode_mode=perplexity.decode_mode,
        config_name=args.config,
    )

    print(f"checkpoint={checkpoint_path}{' (auto-selected)' if auto_selected else ''}")
    print(f"step={checkpoint.get('step')}")
    print(describe_spec(config, prompts))
    print(f"standardized={latent_stats is not None} {resolved.description}")

    metrics, samples, sample_scores = run_perplexity_eval(
        model,
        autoencoder,
        config,
        device,
        dtype,
        latent_stats=latent_stats,
        evaluator=evaluator,
        length_predictor=resolved.predictor,
        predictor_space=resolved.space,
        prompts=prompts,
        return_samples=True,
    )

    if args.baseline:
        from src.train import build_validation_loaders

        texts: list[str] = []
        max_bytes = max((sample.num_bytes for sample in samples), default=512)
        for loader in build_validation_loaders(config).values():
            for batch in loader:
                texts.extend(batch["texts"])
                if len(texts) >= perplexity.baseline_num_texts:
                    break
            if len(texts) >= perplexity.baseline_num_texts:
                break
        metrics.update(score_baseline_texts(evaluator, texts[: perplexity.baseline_num_texts], max_bytes))

    if args.print_samples:
        for index, (sample, score) in enumerate(zip(samples, sample_scores)):
            print()
            print(f"SAMPLE {index + 1} ppl={score['ppl']:.6g} num_tokens={score['num_tokens']:.6g}")
            if sample.prompt is None:
                print(sample.text)
            else:
                print(f"PROMPT       | {sample.prompt}")
                print(f"CONTINUATION | {sample.continuation}")

    print()
    for name, value in metrics.items():
        print(f"{name}={value:.6g}")

    if args.output:
        payload = {
            "checkpoint": checkpoint_path,
            "step": checkpoint.get("step"),
            **metrics,
            "samples": [
                {"prompt": sample.prompt, "text": sample.text, **score}
                for sample, score in zip(samples, sample_scores)
            ],
        }
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
